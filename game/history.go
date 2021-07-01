package game

import (
	"strings"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// HistoryToVariant takes in a game history and returns the board configuration
// and letter distribution name.
func HistoryToVariant(h *pb.GameHistory) (boardLayoutName, letterDistributionName string, variant Variant) {

	boardLayoutName = h.BoardLayout
	// XXX: the letter distribution name should come from the history.
	letterDistributionName = "english"
	switch {
	case strings.HasPrefix(h.Lexicon, "OSPS"):
		letterDistributionName = "polish"
	case strings.HasPrefix(h.Lexicon, "FISE"):
		letterDistributionName = "spanish"
	case strings.HasPrefix(h.Lexicon, "RD"):
		letterDistributionName = "german"
	case strings.HasPrefix(h.Lexicon, "NSF"):
		letterDistributionName = "norwegian"
	case strings.HasPrefix(h.Lexicon, "FRA"):
		letterDistributionName = "french"
	}
	variant = Variant(h.Variant)
	return
}
