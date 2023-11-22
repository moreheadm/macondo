package preendgame

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/kwg"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/tilemapping"
	"github.com/domino14/macondo/zobrist"
	"golang.org/x/sync/errgroup"
)

const InBagMaxLimit = 1
const TieBreakerPlays = 20

type PEGOutcome int

const (
	PEGNotInitialized PEGOutcome = 0
	PEGWin            PEGOutcome = 1
	PEGDraw           PEGOutcome = 2
	PEGLoss           PEGOutcome = 3
)

type Outcome struct {
	tiles   []tilemapping.MachineLetter
	ct      int
	outcome PEGOutcome
}

// Equal tells whether a and b contain the same elements.
// A nil argument is equivalent to an empty slice.
func Equal(a, b []tilemapping.MachineLetter) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

type PreEndgamePlay struct {
	sync.RWMutex
	Play          *move.Move
	Points        float32
	FoundLosses   float32
	Spread        int
	spreadSet     bool
	outcomesArray []Outcome
	Ignore        bool
}

func (p *PreEndgamePlay) stopAnalyzing() {
	p.Lock()
	defer p.Unlock()
	p.Ignore = true
}

func (p *PreEndgamePlay) outcomeIndex(tiles []tilemapping.MachineLetter) int {
	found := -1
	for idx, outcome := range p.outcomesArray {
		if Equal(outcome.tiles, tiles) {
			found = idx
			break
		}
	}
	if found == -1 {
		found = len(p.outcomesArray)
		p.outcomesArray = append(p.outcomesArray, Outcome{
			tiles: tiles,
		})
	}
	return found
}

func (p *PreEndgamePlay) addWinPctStat(result PEGOutcome, ct int, tiles []tilemapping.MachineLetter) {
	p.Lock()
	defer p.Unlock()
	found := p.outcomeIndex(tiles)
	p.outcomesArray[found].outcome = result
	p.outcomesArray[found].ct += ct
	switch result {
	case PEGWin:
		p.Points += float32(ct)
	case PEGDraw:
		p.Points += float32(ct) / 2
		p.FoundLosses += float32(ct) / 2
	case PEGLoss:
		// no wins
		p.FoundLosses += float32(ct)
	}
}

func (p *PreEndgamePlay) addSpreadStat(spread, ct int) {
	p.Lock()
	defer p.Unlock()
	p.spreadSet = true
	p.Spread += (spread * ct)
}

func (p *PreEndgamePlay) setWinPctStat(result PEGOutcome, ct int, tiles []tilemapping.MachineLetter) {
	p.Lock()
	defer p.Unlock()
	found := p.outcomeIndex(tiles)

	// If any draw is found for a combination of tiles, that whole
	// combination gets classified as a draw at best, and a loss at worst.
	// If any loss is found for a combination of tiles, that whole
	// combination is a loss.
	// If any win is found for a combination of tiles, we must make
	// sure that they're ALL wins before calling it a win.
	switch result {
	case PEGWin:
		if p.outcomesArray[found].outcome != PEGDraw &&
			p.outcomesArray[found].outcome != PEGLoss {

			// Add to the win counter only if it wasn't already marked a win.
			// Note that this win is not necessarily known yet.
			if p.outcomesArray[found].outcome != PEGWin {
				p.Points += float32(ct)
			}

			p.outcomesArray[found].outcome = PEGWin

		}
	case PEGDraw:
		if p.outcomesArray[found].outcome != PEGLoss {
			// Add to the win counter only if it wasn't already marked a draw.
			// Note that this draw is not necessarily known yet.

			if p.outcomesArray[found].outcome != PEGDraw {
				p.Points += float32(ct) / 2
				p.FoundLosses += float32(ct) / 2
			}
			p.outcomesArray[found].outcome = PEGDraw

		}
	case PEGLoss:
		if p.outcomesArray[found].outcome == PEGDraw {
			p.Points -= float32(ct) / 2
			p.FoundLosses += float32(ct) / 2
		} else if p.outcomesArray[found].outcome == PEGWin {
			p.Points -= float32(ct)
			p.FoundLosses += float32(ct)
		} else if p.outcomesArray[found].outcome == PEGNotInitialized {
			p.FoundLosses += float32(ct)
		}
		p.outcomesArray[found].outcome = PEGLoss

	}
}

func (p *PreEndgamePlay) HasLoss(tiles []tilemapping.MachineLetter) bool {
	found := -1
	for idx, outcome := range p.outcomesArray {
		if Equal(outcome.tiles, tiles) {
			found = idx
			break
		}
	}
	if found == -1 {
		return false
	}
	p.RLock()
	defer p.RUnlock()
	return p.outcomesArray[found].outcome == PEGLoss
}

func (p *PreEndgamePlay) AllHaveLoss(tiles [][]tilemapping.MachineLetter) bool {
	p.RLock()
	defer p.RUnlock()
	for _, tileset := range tiles {
		found := -1
		for idx, outcome := range p.outcomesArray {
			if Equal(outcome.tiles, tileset) {
				found = idx
				if p.outcomesArray[idx].outcome != PEGLoss {
					return false
				}
				break
			}
		}
		if found == -1 {
			// We have no results for this tileset, so caller must search.
			return false
		}
	}
	return true
}

func (p *PreEndgamePlay) OutcomeFor(tiles []tilemapping.MachineLetter) PEGOutcome {
	found := -1
	for idx, outcome := range p.outcomesArray {
		if Equal(outcome.tiles, tiles) {
			found = idx
			break
		}
	}
	if found == -1 {
		return PEGNotInitialized
	}
	p.RLock()
	defer p.RUnlock()
	return p.outcomesArray[found].outcome
}

func (p *PreEndgamePlay) String() string {
	return fmt.Sprintf("<play %v, wins %f>", p.Play.ShortDescription(), p.Points)
}

type Solver struct {
	endgameSolvers []*negamax.Solver

	movegen       movegen.MoveGenerator
	game          *game.Game
	gaddag        *kwg.KWG
	ttable        *negamax.TranspositionTable
	endgamePlies  int
	initialSpread int
	threads       int
	numinbag      int
	plays         []*PreEndgamePlay
	winnerSoFar   *PreEndgamePlay
	knownOppRack  []tilemapping.MachineLetter
	busy          bool

	earlyCutoffOptim bool
	skipPassOptim    bool
	skipTiebreaker   bool
	skipLossOptim    bool

	numEndgamesSolved  atomic.Uint64
	numCutoffs         atomic.Uint64
	minPotentialLosses float32
}

// Init initializes the solver. It creates all the parallel endgame solvers.
func (s *Solver) Init(g *game.Game, gd *kwg.KWG) error {
	s.ttable = negamax.GlobalTranspositionTable
	s.threads = max(1, runtime.NumCPU())
	s.ttable.SetMultiThreadedMode()
	s.game = g.Copy()
	s.game.SetBackupMode(game.SimulationMode)
	// To control what tiles we draw:
	s.game.Bag().SetFixedOrder(true)
	s.endgamePlies = 4
	s.gaddag = gd
	s.earlyCutoffOptim = true
	s.skipPassOptim = false
	s.skipTiebreaker = false
	return nil
}

func (s *Solver) Solve(ctx context.Context) ([]*PreEndgamePlay, error) {
	s.busy = true
	defer func() {
		s.busy = false
	}()
	log.Info().
		Int("endgame-plies", s.endgamePlies).
		Bool("early-cutoff-optim", s.earlyCutoffOptim).
		Bool("skip-pass-optim", s.skipPassOptim).
		Bool("skip-tiebreaker-optim", s.skipTiebreaker).
		Bool("skip-loss-optim", s.skipLossOptim).
		Int("threads", s.threads).
		Msg("preendgame-solve-called")

	s.numEndgamesSolved.Store(0)
	s.numCutoffs.Store(0)

	s.minPotentialLosses = 100000.0

	playerOnTurn := s.game.PlayerOnTurn()
	// Fill opponent's rack for now. Ignore the "known opp rack", if any. That
	// is handled properly later.
	if s.game.RackFor(1-playerOnTurn).NumTiles() < game.RackTileLimit {
		_, err := s.game.SetRandomRack(1-playerOnTurn, nil)
		if err != nil {
			return nil, err
		}
	}

	if s.game.Bag().TilesRemaining() > InBagMaxLimit {
		return nil, fmt.Errorf("bag has too many tiles remaining; limit is %d", InBagMaxLimit)
	} else if s.game.Bag().TilesRemaining() == 0 {
		return nil, errors.New("bag is empty; use endgame solver instead")
	}
	s.numinbag = s.game.Bag().TilesRemaining()
	s.endgameSolvers = make([]*negamax.Solver, s.threads)
	s.ttable.Reset(s.game.Config().TTableFractionOfMem, s.game.Board().Dim())
	s.initialSpread = s.game.CurrentSpread()

	for idx := range s.endgameSolvers {
		es := &negamax.Solver{}
		// share the same transposition table and zobrist params across all endgames.
		// Copy the game so each endgame solver can manipulate it independently.
		g := s.game.Copy()
		g.SetBackupMode(game.SimulationMode)
		// we need to set the state stack length now to account for the PEG move.
		// there can also be passes etc. just add a hacky number.
		// XXX: state stack length should probably just be a fixed large number.
		g.SetStateStackLength(s.endgamePlies + 5)
		g.SetEndgameMode(true)
		// Set a fixed order for the bag. This makes it easy for us to control
		// what tiles we draw after making a move.
		g.Bag().SetFixedOrder(true)
		mg := movegen.NewGordonGenerator(s.gaddag, g.Board(), g.Bag().LetterDistribution())
		err := es.Init(mg, g)
		if err != nil {
			return nil, err
		}
		// Endgame itself should be single-threaded; we are solving many individual
		// endgames in parallel.
		es.SetThreads(1)

		// Endgame should quit early if it finds any win.
		es.SetFirstWinOptim(true)
		s.endgameSolvers[idx] = es
	}
	s.movegen = movegen.NewGordonGenerator(s.gaddag, s.game.Board(), s.game.Bag().LetterDistribution())
	s.movegen.SetGenPass(true)
	// Don't allow pre-endgame opponent to use more than 7 tiles.
	s.movegen.SetMaxTileUsage(7)
	// Examine high equity plays first.
	moves := s.movegen.GenAll(s.game.RackFor(s.game.PlayerOnTurn()), false)
	c, err := equity.NewCombinedStaticCalculator(
		s.game.LexiconName(), s.game.Config(), "", equity.PEGAdjustmentFilename)
	if err != nil {
		return nil, err
	}
	for _, m := range moves {
		m.SetEquity(c.Equity(m, s.game.Board(), s.game.Bag(), nil))
	}
	sort.Slice(moves, func(i, j int) bool {
		return moves[i].Equity() > moves[j].Equity()
	})

	log.Info().Int("nmoves", len(moves)).Int("nthreads", s.threads).Msg("peg-generated-moves")
	winners, err := s.multithreadSolve(ctx, moves)
	return winners, err
}

type job struct {
	ourMove   *PreEndgamePlay
	theirMove *move.Move
	inbag     []tilemapping.MachineLetter
	numDraws  int // how many ways can inbag be drawn?
	fullSolve bool
}

func moveIsPossible(mtiles []tilemapping.MachineLetter, partialRack []tilemapping.MachineLetter) bool {
	// check whether m is a possible move given the total pool of tiles to choose from
	// (unseenRack) and the known partial rack.
	// Note: assumes that m can be made from unseenRack.
	// For example, the following state is impossible:
	// play: COOKIE
	// partial: KLL
	// bag: CEIKLLOO

	partialCopy := make([]int, zobrist.MaxLetters)
	pcount := 0
	for _, t := range partialRack {
		partialCopy[t]++
		pcount++
	}

	for _, t := range mtiles {
		if t == 0 {
			continue
		}
		t = t.IntrinsicTileIdx()
		if partialCopy[t] > 0 {
			partialCopy[t]--
			pcount--
		}
	}
	// unseen: LL
	// partial: LL
	// Try to re-add the letters to partialRack.
	for _, t := range mtiles {
		if t == 0 {
			continue
		}
		t = t.IntrinsicTileIdx()
		partialCopy[t]++
		pcount++
	}
	return pcount <= game.RackTileLimit
}

func (s *Solver) multithreadSolve(ctx context.Context, moves []*move.Move) ([]*PreEndgamePlay, error) {
	// for every move, solve all the possible endgames.
	// - make play on board
	// - for tile in unseen:
	//   - if we've already seen this letter for this pre-endgame move
	//     increment its stats accordingly
	//   - overwrite letters on both racks accordingly
	//   - solve endgame from opp perspective
	//   - increment wins/losses accordingly for this move and letter
	// at the end sort stats by number of won endgames and then avg spread.

	s.plays = make([]*PreEndgamePlay, len(moves))
	for idx, play := range moves {
		s.plays[idx] = &PreEndgamePlay{Play: play}
	}

	maybeInBagTiles := make([]int, tilemapping.MaxAlphabetSize)
	unseenRack := []tilemapping.MachineLetter{}
	for _, t := range s.game.RackFor(s.game.NextPlayer()).TilesOn() {
		maybeInBagTiles[t]++
		unseenRack = append(unseenRack, t)
	}
	for _, t := range s.game.Bag().Peek() {
		maybeInBagTiles[t]++
		unseenRack = append(unseenRack, t)
	}
	// If we have a partial or full opponent rack, these tiles cannot be in
	// the bag.
	for _, t := range s.knownOppRack {
		maybeInBagTiles[t]--
	}

	g := errgroup.Group{}
	winnerGroup := errgroup.Group{}
	log.Debug().Interface("maybe-in-bag-tiles", maybeInBagTiles).Msg("unseen tiles")
	jobChan := make(chan job, s.threads*2)
	winnerChan := make(chan *PreEndgamePlay)

	var processed atomic.Uint32

	for t := 0; t < s.threads; t++ {
		t := t
		g.Go(func() error {
			for j := range jobChan {
				if err := s.handleJob(ctx, j, t, winnerChan); err != nil {
					log.Debug().AnErr("err", err).Msg("error-handling-job")
					// Don't exit, to avoid deadlock.
				}
				processed.Add(1)
				n := processed.Load()
				if n%500 == 0 {
					log.Info().Uint64("cutoffs", s.numCutoffs.Load()).Msgf("processed %d endgames...", n)
				}
			}
			return nil
		})
	}
	// The determiner of the winner.
	winnerGroup.Go(func() error {
		for p := range winnerChan {
			if s.winnerSoFar != nil {
				if p.Points > s.winnerSoFar.Points {
					s.winnerSoFar = p
				}
			} else {
				s.winnerSoFar = p
			}
			// e.g. if we have three known losses in 4 games, we have at most 7 possible losses.
			ppotentialLosses := 8.0 - p.Points // XXX: obviously need to rewrite for non PEG-1
			if ppotentialLosses < s.minPotentialLosses {
				log.Info().
					Float32("potentialLosses", ppotentialLosses).
					Str("p", p.String()).
					Float32("n", s.minPotentialLosses).Msg("min potential losses")
				s.minPotentialLosses = ppotentialLosses
			}
		}
		return nil
	})
	queuedJobs := 0
	var ourPass *PreEndgamePlay

	for t, count := range maybeInBagTiles {
		if count == 0 {
			continue
		}
		// Help the early cutoff optimization by separating out plays in time.
		// XXX: a better optimization might start with endgames where the
		// opponent has "better" top equity plays. i.e. let them bingo out
		// right off the bat for example so that we can get earlier cutoffs)
		for _, p := range s.plays {
			if p.Play.Action() == move.MoveTypePass {
				// passes handled differently.
				ourPass = p
				continue
			}
			j := job{
				ourMove:  p,
				inbag:    []tilemapping.MachineLetter{tilemapping.MachineLetter(t)},
				numDraws: count,
			}
			queuedJobs++
			jobChan <- j
		}
	}

	if !s.skipPassOptim {
		// Handle pass.
		// First, try to pass back with all possible racks.
		for t, count := range maybeInBagTiles {
			if count == 0 {
				continue
			}
			j := job{
				ourMove:   ourPass,
				theirMove: move.NewPassMove(nil, s.game.Alphabet()),
				inbag:     []tilemapping.MachineLetter{tilemapping.MachineLetter(t)},
				numDraws:  count,
			}
			queuedJobs++
			jobChan <- j
		}

		// Then, for every combination of 7 tiles they could have,
		// generate all plays, make each play, and solve endgame from our
		// perspective.
		theirPossibleRack := tilemapping.NewRack(s.game.Alphabet())
		theirPossibleRack.Set(unseenRack)
		// Generate all possible plays for our opponent that are not the pass
		// back (was just handled above).
		s.movegen.SetGenPass(false)
		theirMoves := s.movegen.GenAll(theirPossibleRack, false)
		s.movegen.SetGenPass(true)
		for _, m := range theirMoves {
			if moveIsPossible(m.Tiles(), s.knownOppRack) {
				j := job{
					ourMove:   ourPass,
					theirMove: m,
					inbag:     unseenRack,
				}
				queuedJobs++
				jobChan <- j
			}
		}
	} else {
		log.Info().Msg("skipping pass analysis")
	}

	log.Info().Int("numJobs", queuedJobs).Msg("queued-jobs")
	close(jobChan)
	err := g.Wait()
	if err != nil {
		return nil, err
	}

	close(winnerChan)
	winnerGroup.Wait()

	// sort plays by win %
	sort.Slice(s.plays, func(i, j int) bool {
		return s.plays[i].Points > s.plays[j].Points
	})

	if !s.skipTiebreaker {
		err = s.maybeTiebreak(ctx, maybeInBagTiles)
		if err != nil {
			return nil, err
		}
	}

	if ctx.Err() != nil && (ctx.Err() == context.Canceled || ctx.Err() == context.DeadlineExceeded) {
		log.Info().Msg("timed out or stopped; returning best results so far...")
	}
	log.Info().Uint64("solved-endgames", s.numEndgamesSolved.Load()).
		Uint64("cutoff-moves", s.numCutoffs.Load()).
		Str("winner", s.plays[0].String()).Msg("winning-play")

	return s.plays, err
}

func (s *Solver) maybeTiebreak(ctx context.Context, maybeInBagTiles []int) error {
	// Now, among all the winners find the top spreads.
	i := 0
	for {
		if i+1 >= len(s.plays) || s.plays[i].Points != s.plays[i+1].Points {
			break
		}
		i++
	}
	if i == 0 {
		log.Info().Str("winner", s.plays[0].String()).Msg("only one clear winner")
		return nil
	}
	numWinners := i + 1

	// We want to solve these endgames fully (to get an accurate spread)
	for _, es := range s.endgameSolvers {
		es.SetFirstWinOptim(false)
	}

	g := errgroup.Group{}
	winnerGroup := errgroup.Group{}
	jobChan := make(chan job, s.threads*2)
	winnerChan := make(chan *PreEndgamePlay)

	for t := 0; t < s.threads; t++ {
		t := t
		g.Go(func() error {
			for j := range jobChan {
				if err := s.handleJob(ctx, j, t, winnerChan); err != nil {
					log.Debug().AnErr("err", err).Msg("error-handling-job")
					// Don't exit, to avoid deadlock.
				}
			}
			return nil
		})
	}
	// The determiner of the winner.
	winnerGroup.Go(func() error {
		for p := range winnerChan {
			if !s.winnerSoFar.spreadSet {
				s.winnerSoFar = p
			} else if p.Spread > s.winnerSoFar.Spread {
				s.winnerSoFar = p
			}

		}
		return nil
	})

	// There is more than one play. Use total points scored as a first tie-breaker
	topPlays := s.plays[:numWinners]
	sort.Slice(topPlays, func(i, j int) bool {
		return topPlays[i].Play.Score() > topPlays[j].Play.Score()
	})
	topN := min(TieBreakerPlays, len(topPlays))
	log.Info().Msgf("%d plays tied for first, taking top %d and tie-breaking...", numWinners, topN)
	topPlays = topPlays[:topN]

	// for simplicity's sake, let's skip the pass if that's one of the top
	// plays. gotta cut corners somewhere.
	queuedJobs := 0
	for _, p := range topPlays {
		if p.Play.Action() == move.MoveTypePass {
			continue
		}
		for t, count := range maybeInBagTiles {
			if count == 0 {
				continue
			}
			j := job{
				ourMove:   p,
				inbag:     []tilemapping.MachineLetter{tilemapping.MachineLetter(t)},
				numDraws:  count,
				fullSolve: true,
			}
			queuedJobs++
			jobChan <- j
		}
	}

	log.Info().Int("numTiebreakerJobs", queuedJobs).Msg("queued-jobs")
	close(jobChan)
	err := g.Wait()
	if err != nil {
		return err
	}

	close(winnerChan)
	winnerGroup.Wait()

	sort.Slice(topPlays, func(i, j int) bool {
		// plays without spread set should be at the bottom.
		if !topPlays[i].spreadSet {
			return false
		}
		if !topPlays[j].spreadSet {
			return true
		}
		return topPlays[i].Spread > topPlays[j].Spread
	})

	return nil
}

func (s *Solver) SetEndgamePlies(p int) {
	s.endgamePlies = p
}

func (s *Solver) SetThreads(t int) {
	s.threads = t
}

func (s *Solver) handleJob(ctx context.Context, j job, thread int, winnerChan chan *PreEndgamePlay) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if s.skipLossOptim {
		j.ourMove.RLock()
		if j.ourMove.FoundLosses > 0 {
			j.ourMove.RUnlock()
			j.ourMove.stopAnalyzing()
			s.numCutoffs.Add(1)
			return nil
		}
		j.ourMove.RUnlock()
	}

	if s.earlyCutoffOptim {
		j.ourMove.RLock()
		// we should check to see if our move has more found losses than
		// _any_ fully analyzed move. If so, it can't possibly win.

		if j.ourMove.FoundLosses > s.minPotentialLosses {
			// cut off this play. We already have more losses than the
			// fully analyzed play with the minimum known number of losses.
			j.ourMove.RUnlock()
			log.Debug().Float32("foundLosses", j.ourMove.FoundLosses).
				Float32("minKnownLosses", s.minPotentialLosses).
				Str("ourMove", j.ourMove.String()).
				Msg("stop-analyzing-move")
			j.ourMove.stopAnalyzing()
			s.numCutoffs.Add(1)
			return nil
		}
		j.ourMove.RUnlock()
	}
	if j.ourMove.Play.Action() == move.MoveTypePass && j.theirMove.Action() != move.MoveTypePass {
		return s.handleNonpassResponseToPass(ctx, j, thread)
	}

	g := s.endgameSolvers[thread].Game()
	// throw opponent's rack in, and then enforce a tile order.
	g.ThrowRacksInFor(1 - g.PlayerOnTurn())
	// Basically, put the tiles we want to draw on the right side of the bag.
	// The bag drawing algorithm draws tiles from right to left. We put the
	// "inbag" tiles to the left/"beginning" of the bag so that we can't draw them.
	moveTilesToBeginning(j.inbag, g.Bag())
	// And redraw tiles for opponent. Note that this is not an actual
	// random rack! We are choosing which tiles to draw via the
	// moveTilesToBeginning call above and the fixedOrder setting for the bag.
	_, err := g.SetRandomRack(1-g.PlayerOnTurn(), nil)
	if err != nil {
		return err
	}
	// Finally play move.
	// log.Debug().Interface("drawnLetters",
	// 	Int("ct", j.numDraws).
	// 	Int("thread", thread).
	// 	Str("rack-for-us", g.RackLettersFor(g.PlayerOnTurn())).
	// 	Str("rack-for-them", g.RackLettersFor(1-g.PlayerOnTurn())).
	// 	Str("play", j.ourMove.play.ShortDescription()).Msg("trying-peg-play")

	err = g.PlayMove(j.ourMove.Play, false, 0)
	if err != nil {
		return err
	}
	var finalSpread int16
	// Handle the two-pass situation:
	if j.ourMove.Play.Action() == move.MoveTypePass {
		// Just try a pass from opponent's perspective. This should end the game.
		if j.theirMove.Action() != move.MoveTypePass {
			// we already handled this upon entering this function.
			panic("unexpected move type for them")
		}
		err = g.PlayMove(j.theirMove, false, 0)
		if err != nil {
			return err
		}
		if g.Playing() != macondo.PlayState_GAME_OVER {
			panic("unexpected game is still ongoing")
		}
		// finalSpread should be calculated from our opponent's perspective
		// for the below checks.
		finalSpread = -int16(g.CurrentSpread())
		// Unplay opponent's pass
		g.UnplayLastMove()

	} else {

		// This is the spread after we make our play, from the POV of our
		// opponent.
		initialSpread := g.CurrentSpread()
		// Now let's solve the endgame for our opponent.
		val, _, err := s.endgameSolvers[thread].QuickAndDirtySolve(ctx, s.endgamePlies, thread)
		if err != nil {
			return err
		}
		s.numEndgamesSolved.Add(1)

		// val is the gain in spread after endgame (or loss, if negative), from
		// POV of opponent.
		// so the actual final spread is val + initialSpread
		finalSpread = val + int16(initialSpread)

	}
	if !j.fullSolve {
		switch {

		case finalSpread > 0:
			// win for our opponent = loss for us
			log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msg("we-lose")
			j.ourMove.addWinPctStat(PEGLoss, j.numDraws, j.inbag)
		case finalSpread == 0:
			// draw
			log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msg("we-tie")
			j.ourMove.addWinPctStat(PEGDraw, j.numDraws, j.inbag)
		case finalSpread < 0:
			// loss for our opponent = win for us
			log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msg("we-win")
			j.ourMove.addWinPctStat(PEGWin, j.numDraws, j.inbag)
		}
	} else {
		// If it's a full solve, we actually care about the value of
		// finalSpread. Negate it because this spread is from the POV of
		// our opponent:
		j.ourMove.addSpreadStat(int(-finalSpread), j.numDraws)
		log.Debug().Str("move", j.ourMove.String()).
			Str("inbag", tilemapping.MachineWord(j.inbag).UserVisible(j.ourMove.Play.Alphabet())).
			Int("spread", -int(finalSpread)).Int("ndraws", j.numDraws).Msg("adding-spread-stat")

	}

	log.Debug().Int("thread", thread).Msg("peg-unplay")

	g.UnplayLastMove()
	winnerChan <- j.ourMove
	return nil
}

func (s *Solver) handleNonpassResponseToPass(ctx context.Context, j job, thread int) error {
	// This function handles a situation in the pre-endgame where we start with
	// a pass, but opponent makes a play that is not a pass.

	// - we have unseen tiles (j.inbag, not a great name for this variable)
	// and tiles in the play (j.theirMove)
	// - determine which tiles could be in the bag and still allow j.theirMove to
	// be played. Call this set <T>
	// - If for EVERY tileset in <T> we have a result of LOSS already, we can exit
	// early.
	// - If for ANY tileset in <T> we have a result of DRAW, we must still analyze
	// to make sure this isn't a loss.
	// - If for ANY tileset in <T> we have a result of WIN, we must still analyze
	// to make sure this isn't a draw or loss.
	// clean this up, way too inefficient.
	pt := possibleTilesInBag(j.inbag, j.theirMove.Tiles(), s.knownOppRack)
	// XXX: 1-PEG. Change this when we support 2-PEG (and beyond?)
	if len(pt) == 0 {
		log.Warn().Msgf("possible tiles in bag is empty; inbag = %v, theirMove = %v, oppRack = %v",
			j.inbag, j.theirMove, s.knownOppRack)
		return nil
	}
	splitPt := permuteLeaves(pt, 1)

	if j.ourMove.AllHaveLoss(splitPt) {
		log.Debug().Str("their-move", j.theirMove.ShortDescription()).
			Msg("exiting-early-no-new-info")
		return nil
	}

	g := s.endgameSolvers[thread].Game()

	// throw opponent's rack in
	g.ThrowRacksInFor(1 - g.PlayerOnTurn())
	rack := tilemapping.NewRack(g.Alphabet())
	rack.Set(j.inbag)
	// Assign opponent the entire rack, which may be longer than 7 tiles long.
	g.SetRackForOnly(1-g.PlayerOnTurn(), rack)

	log.Debug().Interface("drawnLetters",
		tilemapping.MachineWord(j.inbag).UserVisible(g.Alphabet())).
		Int("ct", j.numDraws).
		Int("thread", thread).
		Str("rack-for-us", g.RackLettersFor(g.PlayerOnTurn())).
		Str("rack-for-them", g.RackLettersFor(1-g.PlayerOnTurn())).
		Str("their-play", j.theirMove.ShortDescription()).
		Msgf("trying-peg-play; splitpt=%v", splitPt)

	// Play our pass
	err := g.PlayMove(j.ourMove.Play, false, 0)
	if err != nil {
		return err
	}
	// Play their play
	err = g.PlayMove(j.theirMove, false, 0)
	if err != nil {
		return err
	}
	// solve the endgame from OUR perspective
	// This is the spread for us currently.
	initialSpread := g.CurrentSpread()
	val, _, err := s.endgameSolvers[thread].QuickAndDirtySolve(ctx, s.endgamePlies, thread)
	if err != nil {
		return err
	}
	s.numEndgamesSolved.Add(1)

	// val is the gain in spread after endgame (or loss, if negative), from
	// our own POV.
	// so the actual final spread is val + initialSpread
	finalSpread := val + int16(initialSpread)

	for _, tileset := range splitPt {
		ct := 0
		for _, t := range j.inbag {
			// XXX: assumes 1-PEG. Rework this later.
			if tileset[0] == t {
				ct++
			}
		}

		switch {

		case finalSpread > 0:
			// win for us
			log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msgf("p-we-win-tileset-%v", tileset)
			j.ourMove.setWinPctStat(PEGWin, ct, tileset)
		case finalSpread == 0:
			// draw
			log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msgf("p-we-tie-tileset-%v", tileset)
			j.ourMove.setWinPctStat(PEGDraw, ct, tileset)
		case finalSpread < 0:
			// loss for us
			log.Debug().Int16("finalSpread", finalSpread).Int("thread", thread).Msgf("p-we-lose-tileset-%v", tileset)
			j.ourMove.setWinPctStat(PEGLoss, ct, tileset)
		}
	}

	g.UnplayLastMove() // Unplay opponent's last move.
	g.UnplayLastMove() // and unplay the pass from our end that started this whole thing.

	return nil
}

func moveTilesToBeginning(order []tilemapping.MachineLetter, bag *tilemapping.Bag) {
	bagContents := bag.Tiles()
	lastPlacedTile := 0
	for didx := lastPlacedTile; didx < len(order); didx++ {
		for bidx, bagTile := range bagContents {
			place := len(order) - didx - 1
			desiredTile := order[didx]
			if desiredTile == bagTile {
				bag.SwapTile(place, bidx)
				lastPlacedTile++
				break
			}
		}
	}
}

func possibleTilesInBag(unseenTiles []tilemapping.MachineLetter, moveTiles []tilemapping.MachineLetter,
	knownPlayerTiles []tilemapping.MachineLetter) []tilemapping.MachineLetter {
	// given a set of unseenTiles and a move, determine which of the unseen tiles
	// could be in the bag and still allow for move to be played.
	// Return a deduplicated set, essentially.

	ucTally := make([]tilemapping.MachineLetter, zobrist.MaxLetters)
	for _, u := range unseenTiles {
		ucTally[u]++
	}
	kpTally := make([]tilemapping.MachineLetter, zobrist.MaxLetters)
	for _, t := range knownPlayerTiles {
		kpTally[t]++
	}

	// ucTally contains tiles that are unseen and not in the move. However,
	// if we know that some of these tiles are in our hand, they cannot
	// possibly be in the bag.
	for _, t := range moveTiles {
		if t == 0 {
			continue // not a played tile
		}
		t = t.IntrinsicTileIdx()
		for _, u := range unseenTiles {
			if t == u {
				// First take it away from the known player tiles, if there
				if kpTally[u] > 0 {
					kpTally[u]--
				}
				ucTally[u]--
				break
			}
		}
	}
	for idx := range kpTally {
		ucTally[idx] -= kpTally[idx]
	}

	pt := []tilemapping.MachineLetter{}
	for idx, u := range ucTally {
		if u > 0 {
			pt = append(pt, tilemapping.MachineLetter(idx))
		}
	}
	return pt
}

func (s *Solver) SolutionStats(maxMoves int) string {
	// Assume plays are already sorted.
	var ss strings.Builder
	fmt.Fprintf(&ss, "%-20s%-5s%-7s%-9s%-32s%-2s\n", "Play", "Wins", "%Win", "Spread", "Outcomes", "")

	for _, play := range s.plays[:maxMoves] {
		noutcomes := 0
		for _, el := range play.outcomesArray {
			noutcomes += el.ct
		}

		ignore := ""
		wpStats := "---"
		pts := "---"
		spdStats := ""
		if play.Ignore {
			ignore = "❌"
		} else {
			wpStats = fmt.Sprintf("%.2f", 100.0*play.Points/float32(noutcomes))
			pts = fmt.Sprintf("%.1f", play.Points)
			if play.spreadSet {
				spdStats = fmt.Sprintf("%.2f", float32(play.Spread)/float32(noutcomes))
			}
		}
		var wins, draws, losses [][]tilemapping.MachineLetter
		var outcomeStr string
		for _, outcome := range play.outcomesArray {
			// uv := tilemapping.MachineWord(outcome.tiles).UserVisible(s.game.Alphabet())
			if outcome.outcome == PEGWin {
				wins = append(wins, outcome.tiles)
			} else if outcome.outcome == PEGDraw {
				draws = append(draws, outcome.tiles)
			} else if outcome.outcome == PEGLoss {
				losses = append(losses, outcome.tiles)
			}
			// otherwise, it's unknown/not calculated.
		}
		if len(wins) > 0 {
			outcomeStr += fmt.Sprintf("👍: %s ", toUserFriendly(wins, s.game.Alphabet()))
		}
		if len(draws) > 0 {
			outcomeStr += fmt.Sprintf("🤝: %s ", toUserFriendly(draws, s.game.Alphabet()))
		}
		if len(losses) > 0 {
			outcomeStr += fmt.Sprintf("👎: %s", toUserFriendly(losses, s.game.Alphabet()))
		}

		fmt.Fprintf(&ss, "%-20s%-5s%-7s%-9s%-32s%-2s\n", play.Play.ShortDescription(),
			pts, wpStats, spdStats, outcomeStr, ignore)
	}
	fmt.Fprintf(&ss, "❌ marks plays cut off early\n")

	return ss.String()
}

func (s *Solver) SetEarlyCutoffOptim(o bool) {
	s.earlyCutoffOptim = o
}

func (s *Solver) SetSkipPassOptim(o bool) {
	s.skipPassOptim = o
}

func (s *Solver) SetSkipLossOptim(o bool) {
	s.skipLossOptim = o
}

func (s *Solver) SetKnownOppRack(rack tilemapping.MachineWord) {
	s.knownOppRack = rack
}

func (s *Solver) SetSkipTiebreaker(o bool) {
	s.skipTiebreaker = o
}

func (s *Solver) IsSolving() bool {
	return s.busy
}

func toUserFriendly(tilesets [][]tilemapping.MachineLetter, alphabet *tilemapping.TileMapping) string {
	var ss strings.Builder
	sort.Slice(tilesets, func(i, j int) bool {
		// XXX: assumes tilesets are the same size.
		for idx := range tilesets[i] {
			if tilesets[i][idx] < tilesets[j][idx] {
				return true
			}
		}
		return false
	})

	for _, s := range tilesets {
		ss.WriteString(tilemapping.MachineWord(s).UserVisible(alphabet))
		ss.WriteString(" ")
	}
	return ss.String()
}
