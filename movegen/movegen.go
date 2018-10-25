// Package movegen contains all the move-generating functions. It makes
// heavy use of the GADDAG.
// Implementation notes:
// Using Gordon's GADDAG algorithm with some minor speed changes. Similar to
// the A&J DAWG algorithm, we should not be doing "for each letter allowed
// on this square", as that is a for loop through every letter in the rack.
// Instead, we should go through the node siblings in the Gen algorithm,
// and check their presence in the cross-sets.
package movegen

import (
	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
)

type MoveType uint8

const (
	MoveTypePlay MoveType = iota
	MoveTypeExchange
	MoveTypePass
	MoveTypePhonyTilesReturned

	MoveTypeEndgameTiles
	MoveTypeLostTileScore
)

// LettersRemain returns true if there is at least one letter in the
// rack, 0 otherwise.
func LettersRemain(rack []uint8) bool {
	for i := 0; i < alphabet.MaxAlphabetSize; i++ {
		if rack[i] > 0 {
			return true
		}
	}
	return false
}

type Move struct {
	action   MoveType
	score    int8
	desc     string
	word     alphabet.MachineWord
	rowStart uint8
	colStart uint8
	vertical bool
	bingo    bool
}

type GordonGenerator struct {
	gaddag   gaddag.SimpleGaddag
	board    GameBoard
	vertical bool // Are we generating moves vertically or not?
	// The move generator works by generating moves starting at an anchor
	// square. curAnchorRow and curAnchorCol are the 0-based coordinates
	// of the current anchor square.
	curAnchorRow int8
	curAnchorCol int8
	tilesPlayed  uint8
	plays        []Move
}

func (gen *GordonGenerator) GenAll(rack []string, board GameBoard) {
	gen.board = board
	gen.curAnchorRow = 7
	gen.curAnchorCol = 7
}

// NextNodeIdx is analogous to NextArc in the Gordon paper. The main difference
// is that in Gordon, the initial state is an arc pointing to the first
// node. In our implementation of the GADDAG, the initial state is that
// first node. So we have to think in terms of the node that was pointed
// to, rather than the pointing arc. There is something slightly wrong with
// the paper as it does not seem possible to implement in exactly Gordon's way
// without running into issues. (See my notes in my `ujamaa` repo in gaddag.h)
// Note: This is a non-deterministic algorithm. However, using a 2-D table
// of nodes/arcs did not speed it up (actually it might have even been slower)
// This is probably due to larger memory usage being cache-inefficient.
func (gen GordonGenerator) NextNodeIdx(nodeIdx uint32, letter alphabet.MachineLetter) uint32 {
	arcs := uint32(gen.gaddag.NumArcs(nodeIdx))
	if arcs == 0 {
		return 0
	}
	for i := nodeIdx + 1; i <= nodeIdx+arcs; i++ {
		idx, nextLetter := gen.gaddag.ArcToIdxLetter(gen.gaddag.Nodes[i])
		if nextLetter == letter {
			return idx
		}
		if nextLetter > letter {
			// Since it's sorted numerically we know it won't be in the arc
			// list, so exit the loop early.
			return 0
		}
	}
	return 0
}

func crossAllowed(cross uint64, letter alphabet.MachineLetter) bool {
	return cross&(1<<uint8(letter)) != 0
}

// Gen is an implementation of the Gordon Gen function.
// pos is the offset from an anchor square.
func (gen *GordonGenerator) Gen(pos int8, word alphabet.MachineWord, rack *Rack,
	nodeIdx uint32) {

	curRow := gen.curAnchorRow
	curCol := gen.curAnchorCol

	var crossSet uint64

	if gen.vertical {
		curRow += pos
	} else {
		curCol += pos
	}

	// If a letter L is already on this square, then GoOn...
	curSquare := gen.board[curRow][curCol]
	curLetter := curSquare.letter

	if gen.vertical {
		crossSet = curSquare.hcrossSet
	} else {
		crossSet = curSquare.vcrossSet
	}

	if curLetter != EmptySquareMarker {
		nnIdx := gen.NextNodeIdx(nodeIdx, curLetter)
		if nnIdx != 0 {
			gen.GoOn(pos, curLetter, word, nnIdx)
		}
	} else if !rack.empty {
		// Instead of doing the loop in the Gordon Gen algorithm, we should
		// just go through the node's children and test them independently
		// against the cross set. Note that some of these children could be
		// the SeparationToken
		arcs := uint32(gen.gaddag.NumArcs(nodeIdx))
		for i := nodeIdx + 1; i <= nodeIdx+arcs; i++ {
			nnIdx, nextLetter := gen.gaddag.ArcToIdxLetter(gen.gaddag.Nodes[i])
			if nextLetter == alphabet.MaxAlphabetSize {
				// The Separation token.
				break
			}
			// The letter must be on the rack AND it must be allowed in the
			// cross-set.
			if !(rack.contains(nextLetter) && crossAllowed(crossSet, nextLetter)) {
				continue
			}
			rack.take(nextLetter)
			gen.tilesPlayed++
			gen.GoOn(pos, nextLetter, word, nnIdx)
			rack.add(nextLetter)
			gen.tilesPlayed--
		}
		// Check for the blanks meow.
		if rack.contains(BlankPos) {
			// Just go through all the children; they're all acceptable if they're
			// in the cross-set.
			for i := nodeIdx + 1; i <= nodeIdx+arcs; i++ {
				nnIdx, nextLetter := gen.gaddag.ArcToIdxLetter(gen.gaddag.Nodes[i])
				if nextLetter == alphabet.MaxAlphabetSize {
					// The separation token
				}
				if !crossAllowed(crossSet, nextLetter) {
					continue
				}
				rack.take(BlankPos)
				gen.tilesPlayed++
				gen.GoOn(pos, nextLetter.Blank(), word, nnIdx)
				rack.add(BlankPos)
				gen.tilesPlayed--

			}
		}
	}

}

func (gen *GordonGenerator) GoOn(pos int8, L alphabet.MachineLetter, word alphabet.MachineWord,
	NewNodeIdx uint32) {

	if pos <= 0 {
		curRow := gen.curAnchorRow
		curCol := gen.curAnchorCol
		var leftRow, leftCol int8
		if gen.vertical {
			curRow += pos
			leftRow = curRow - 1
		} else {
			curCol += pos
			leftCol = curCol - 1
		}

		word = alphabet.MachineWord(L) + word
		// if L on OldArc and no letter directly left, then record play.
		letterDirectlyLeft := false
		// roomToLeft is true unless we are right at the edge of the board.
		roomToLeft := true

		// Check to see if there is a letter directly to the left.
		if leftRow >= 0 && leftCol >= 0 {
			if gen.board[leftRow][leftCol].letter != EmptySquareMarker {
				// There is a letter to the left.
				letterDirectlyLeft = true
			}
		}

		if gen.gaddag.InLetterSet(L, NewNodeIdx) && !letterDirectlyLeft {
			gen.RecordPlay(word)
		}

	} else {

	}
}

func (gen *GordonGenerator) RecordPlay(word alphabet.MachineWord) {
	play := Move{
		action:   MoveTypePlay,
		score:    17,
		desc:     "foo",
		word:     word,
		vertical: gen.vertical,
		bingo:    gen.tilesPlayed == 7,
	}
	gen.plays = append(gen.plays, play)
}

// For future?: The Gordon GADDAG algorithm is somewhat inefficient because
// it goes through all letters on the rack. Then for every letter, it has to
// call the NextNodeIdx or similar function above, which has for loops that
// search for the next child.
// Instead, we need a data structure where the nodes have pointers to their
// "children" or "siblings" on the arcs; we then iterate through all the
// "siblings" and see if their letters are on the rack. This should be
// significantly faster if the data structure is fast.
