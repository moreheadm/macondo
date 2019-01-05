// Package anagrammer uses a DAWG instead of a GADDAG to simplify the
// algorithm and make it potentially faster - we don't need a GADDAG
// to generate anagrams/subanagrams.
//
// This package generates anagrams and subanagrams and has an RPC
// interface.
package anagrammer

import (
	"log"
	"os"
	"strings"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"
	"github.com/domino14/macondo/movegen"
)

func LoadDawgs(dawgPath string) {
	// Load the DAWGs into memory.
	lexica := []string{"America", "CSW15", "FISE09", "osps38"}
	Dawgs = make(map[string]*gaddag.SimpleGaddag)
	for _, lex := range lexica {
		filename := dawgPath + lex + ".dawg"
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			log.Println("[ERROR] File", filename, "did not exist. Continuing...")
			continue
		}

		Dawgs[lex] = gaddag.LoadGaddag(filename)
	}
}

const BlankPos = alphabet.MaxAlphabetSize

type AnagramMode int

const (
	ModeBuild AnagramMode = iota
	ModeExact
	ModePattern
)

var Dawgs map[string]*gaddag.SimpleGaddag

type AnagramStruct struct {
	answerList []string
	mode       AnagramMode
	numLetters int
}

func Anagram(letters string, gd *gaddag.SimpleGaddag, mode AnagramMode) []string {

	letters = strings.ToUpper(letters)
	answerList := []string{}
	runes := []rune(letters)
	rack := movegen.RackFromString(letters, gd.GetAlphabet())

	ahs := &AnagramStruct{
		answerList: answerList,
		mode:       mode,
		numLetters: len(runes),
	}
	stopChan := make(chan struct{})

	go func() {
		anagram(ahs, gd, gd.GetRootNodeIndex(), "", rack)
		close(stopChan)
	}()
	<-stopChan

	return dedupeAndTransformAnswers(ahs.answerList, gd.GetAlphabet())
	//return ahs.answerList
}

func dedupeAndTransformAnswers(answerList []string, alph *alphabet.Alphabet) []string {
	// Use a map to throw away duplicate answers (can happen with blanks)
	// This seems to be significantly faster than allowing the anagramming
	// goroutine to write directly to a map.
	empty := struct{}{}
	answers := make(map[string]struct{})
	for _, answer := range answerList {
		answers[alphabet.MachineWord(answer).UserVisible(alph)] = empty
	}

	// Turn the answers map into a string array.
	answerStrings := make([]string, len(answers))
	i := 0
	for k := range answers {
		answerStrings[i] = k
		i++
	}
	return answerStrings
}

func anagramHelper(letter alphabet.MachineLetter, gd *gaddag.SimpleGaddag,
	ahs *AnagramStruct, nodeIdx uint32, answerSoFar string, rack *movegen.Rack) {

	var nextNodeIdx uint32
	var nextLetter alphabet.MachineLetter

	if gd.InLetterSet(letter, nodeIdx) {
		toCheck := answerSoFar + string(letter)
		if ahs.mode == ModeBuild || (ahs.mode == ModeExact &&
			len(toCheck) == ahs.numLetters) {

			ahs.answerList = append(ahs.answerList, toCheck)
		}
	}

	numArcs := gd.NumArcs(nodeIdx)
	for i := byte(1); i <= numArcs; i++ {
		nextNodeIdx, nextLetter = gd.ArcToIdxLetter(nodeIdx + uint32(i))
		if letter == nextLetter {
			anagram(ahs, gd, nextNodeIdx, answerSoFar+string(letter), rack)
		}
	}
}

func anagram(ahs *AnagramStruct, gd *gaddag.SimpleGaddag, nodeIdx uint32,
	answerSoFar string, rack *movegen.Rack) {

	for idx, val := range rack.LetArr {
		if val == 0 {
			continue
		}
		rack.LetArr[idx]--
		if idx == BlankPos {
			nlet := alphabet.MachineLetter(gd.GetAlphabet().NumLetters())
			for i := alphabet.MachineLetter(0); i < nlet; i++ {
				anagramHelper(i, gd, ahs, nodeIdx, answerSoFar, rack)
			}
		} else {
			letter := alphabet.MachineLetter(idx)
			anagramHelper(letter, gd, ahs, nodeIdx, answerSoFar, rack)
		}

		rack.LetArr[idx]++
	}
}
