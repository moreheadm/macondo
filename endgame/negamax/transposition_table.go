package negamax

import (
	"math"
	"runtime"
	"sync/atomic"

	"github.com/pbnjay/memory"
	"github.com/rs/zerolog/log"
)

const (
	TTExact = 0x01
	TTLower = 0x02
	TTUpper = 0x03
)

const entrySize = 16

const bottom3ByteMask = (1 << 24) - 1
const depthMask = (1 << 6) - 1

// 16 bytes (entrySize)
type TableEntry struct {
	// Don't store the full hash, but the top 5 bytes. The bottom 3 bytes
	// can be determined from the bucket in the array.
	top4bytes    uint32
	score        int16
	fifthbyte    uint8
	flagAndDepth uint8
	play         TinyMove
}

// fullHash calculates the full 64-bit hash for this table entry, given the bottom
// bytes in zval.
func (t TableEntry) fullHash(idx uint64) uint64 {
	return uint64(t.top4bytes)<<32 + uint64(t.fifthbyte)<<24 + (idx & bottom3ByteMask)
}

func (t TableEntry) flag() uint8 {
	return t.flagAndDepth >> 6
}

func (t TableEntry) depth() uint8 {
	return t.flagAndDepth & depthMask
}

func (t TableEntry) valid() bool {
	// a table flag is 1, 2, or 3.
	return t.flag() != 0
}

func (t TableEntry) move() TinyMove {
	return t.play
}

type TableLock interface {
	Lock()
	Unlock()
	RLock()
	RUnlock()
}

type FakeLock struct{}

func (f FakeLock) Lock()    {}
func (f FakeLock) Unlock()  {}
func (f FakeLock) RLock()   {}
func (f FakeLock) RUnlock() {}

type TranspositionTable struct {
	TableLock
	table        []TableEntry
	created      atomic.Uint64
	lookups      atomic.Uint64
	hits         atomic.Uint64
	sizePowerOf2 int
	sizeMask     int
	// "type 2" collisions. A type 2 collision happens when two positions share
	// the same lower bytes. A type 1 collision happens when two positions share the
	// same overall hash. We don't have a super easy way to detect the latter,
	// but it should be much less common.
	t2collisions atomic.Uint64
}

// var globalTranspositionTable DebugTranspositionTable

func (t *TranspositionTable) SetSingleThreadedMode() {
	t.TableLock = &FakeLock{}
}

func (t *TranspositionTable) SetMultiThreadedMode() {
	// t.TableLock = new(sync.RWMutex)
	t.TableLock = &FakeLock{}
}

func (t *TranspositionTable) lookup(zval uint64) TableEntry {
	t.RLock()
	defer t.RUnlock()
	t.lookups.Add(1)
	idx := zval & uint64(t.sizeMask)
	fullHash := t.table[idx].fullHash(idx)
	if fullHash != zval {
		if t.table[idx].valid() {
			// There is another unrelated node at this position.
			t.t2collisions.Add(1)
		}
		return TableEntry{}
	}
	t.hits.Add(1)
	// otherwise, assume the same zobrist hash is the same position. this fails
	// very, very rarely. but it could happen.
	return t.table[idx]
}

func (t *TranspositionTable) store(zval uint64, tentry TableEntry) {
	idx := zval & uint64(t.sizeMask)
	tentry.top4bytes = uint32(zval >> 32)
	tentry.fifthbyte = uint8(zval >> 24)
	t.Lock()
	defer t.Unlock()
	// just overwrite whatever is there for now.
	t.table[idx] = tentry
	t.created.Add(1)
}

func (t *TranspositionTable) Reset(fractionOfMemory float64) {
	t.Lock()
	defer t.Unlock()
	totalMem := memory.TotalMemory()
	desiredNElems := fractionOfMemory * (float64(totalMem) / float64(entrySize))
	// find biggest power of 2 lower than desired.
	t.sizePowerOf2 = int(math.Log2(desiredNElems))
	// Guarantee at least 2^24 elements in the table. Anything less and our
	// 5-byte full hash proxy won't work.
	if t.sizePowerOf2 < 24 {
		t.sizePowerOf2 = 24
	}

	t.sizeMask = ((1 << t.sizePowerOf2) - 1)

	numElems := int(math.Pow(2, float64(t.sizePowerOf2)))

	log.Info().Int("num-elems", numElems).
		Float64("desired-num-elems", desiredNElems).
		Int("estimated-total-memory-bytes", numElems*entrySize).
		Msg("transposition-table-size")
	t.table = nil
	runtime.GC() // ?

	t.table = make([]TableEntry, numElems)
	t.created.Store(0)
	t.lookups.Store(0)
	t.hits.Store(0)
	t.t2collisions.Store(0)
	log.Info().Msg("allocated-transposition-table")
}

// a debug tt

type DebugTableEntry struct {
	score float32
	flag  uint8
	depth uint8
}

type DebugTranspositionTable struct {
	table   map[string]*DebugTableEntry
	created uint64
	lookups uint64
	hits    uint64
}

func (t *DebugTranspositionTable) lookup(cgp string) *DebugTableEntry {
	t.lookups++
	entry := t.table[cgp]
	if entry != nil {
		t.hits++
	}
	return entry
}

func (t *DebugTranspositionTable) store(cgp string, tentry DebugTableEntry) {
	// just overwrite whatever is there for now.
	t.table[cgp] = &tentry
	t.created++
}

func (t *DebugTranspositionTable) reset() {
	t.table = nil
	runtime.GC() // ?
	t.table = make(map[string]*DebugTableEntry)
	t.created = 0
	log.Info().Msg("allocated-debug-transposition-table")
}
