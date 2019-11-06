package alphabeta

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/domino14/macondo/mechanics"
	"github.com/domino14/macondo/movegen"
	"github.com/domino14/macondo/strategy"
)

var LexiconDir = os.Getenv("LEXICON_DIR")

func TestMain(m *testing.M) {
	if _, err := os.Stat("/tmp/gen_america.gaddag"); os.IsNotExist(err) {
		gaddagmaker.GenerateGaddag(filepath.Join(LexiconDir, "America.txt"), true, true)
		os.Rename("out.gaddag", "/tmp/gen_america.gaddag")
	}
	os.Exit(m.Run())
}

func TestSolveComplex(t *testing.T) {
	gd, _ := gaddag.LoadGaddag("/tmp/gen_america.gaddag")
	dist := alphabet.EnglishLetterDistribution()

	game := &mechanics.XWordGame{}
	game.Init(gd, dist)

	generator := movegen.NewGordonGenerator(
		// The strategy doesn't matter right here
		gd, game.Bag(), game.Board(), &strategy.NoLeaveStrategy{},
	)
	alph := game.Alphabet()
	// XXX: Refactor this; we should have something like:
	// game.LoadFromGCG(path, turnnum)
	// That should set the board, the player racks, scores, etc - the whole state
	// Instead we have to do this manually here:
	generator.SetBoardToGame(alph, board.VsRoy)
	s := new(Solver)
	s.Init(generator, game)
	ourRack := alphabet.RackFromString("EFHIKOQ", alph)
	theirRack := alphabet.RackFromString("WZ", alph)
	// game.
}
