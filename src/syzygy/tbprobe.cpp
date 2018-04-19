/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (c) 2013 Ronald de Man
  Copyright (C) 2016-2018 Marco Costalba, Lucas Braesch

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>   // For std::memset and std::memcpy
#include <deque>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <type_traits>

#include "../bitboard.h"
#include "../movegen.h"
#include "../position.h"
#include "../search.h"
#include "../thread_win32.h"
#include "../types.h"
#include "../uci.h"

#include "tbprobe.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#else
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

using namespace Tablebases;

int Tablebases::MaxCardinality;

namespace {

constexpr int TBPIECES = 6; // Max number of supported pieces

enum { BigEndian, LittleEndian };
enum TBType { KEY, WDL, DTZ }; // Used as template parameter

// Each table has a set of flags: all of them refer to DTZ tables, the last one to WDL tables
enum TBFlag { STM = 1, Mapped = 2, WinPlies = 4, LossPlies = 8, SingleValue = 128 };

inline WDLScore operator-(WDLScore d) { return WDLScore(-int(d)); }
inline Square operator^=(Square& s, int i) { return s = Square(int(s) ^ i); }
inline Square operator^(Square s, int i) { return Square(int(s) ^ i); }

const std::string PieceToChar = " PNBRQK  pnbrqk";

int MapPawns[SQUARE_NB];
int MapB1H1H7[SQUARE_NB];
int MapA1D1D4[SQUARE_NB];
int MapKK[10][SQUARE_NB]; // [MapA1D1D4][SQUARE_NB]

int Binomial[6][SQUARE_NB];    // [k][n] k elements from a set of n elements
int LeadPawnIdx[5][SQUARE_NB]; // [leadPawnsCnt][SQUARE_NB]
int LeadPawnsSize[5][4];       // [leadPawnsCnt][FILE_A..FILE_D]

// Comparison function to sort leading pawns in ascending MapPawns[] order
bool pawns_comp(Square i, Square j) { return MapPawns[i] < MapPawns[j]; }
int off_A1H8(Square sq) { return int(rank_of(sq)) - file_of(sq); }

constexpr Value WDL_to_value[] = {
   -VALUE_MATE + MAX_PLY + 1,
    VALUE_DRAW - 2,
    VALUE_DRAW,
    VALUE_DRAW + 2,
    VALUE_MATE - MAX_PLY - 1
};

template<typename T, int Half = sizeof(T) / 2, int End = sizeof(T) - 1>
inline void swap_endian(T& x)
{
    static_assert(std::is_unsigned<T>::value, "Argument of swap_endian not unsigned");

    uint8_t tmp, *c = (uint8_t*)&x;
    for (int i = 0; i < Half; ++i)
        tmp = c[i], c[i] = c[End - i], c[End - i] = tmp;
}
template<> inline void swap_endian<uint8_t>(uint8_t&) {}

template<typename T, int LE> T number(void* addr)
{
    static const union { uint32_t i; char c[4]; } Le = { 0x01020304 };
    static const bool IsLittleEndian = (Le.c[0] == 4);

    T v;

    if ((uintptr_t)addr & (alignof(T) - 1)) // Unaligned pointer (very rare)
        std::memcpy(&v, addr, sizeof(T));
    else
        v = *((T*)addr);

    if (LE != IsLittleEndian)
        swap_endian(v);
    return v;
}

// DTZ tables don't store valid scores for moves that reset the rule50 counter
// like captures and pawn moves but we can easily recover the correct dtz of the
// previous move if we know the position's WDL score.
int dtz_before_zeroing(WDLScore wdl) {
    return wdl == WDLWin         ?  1   :
           wdl == WDLCursedWin   ?  101 :
           wdl == WDLBlessedLoss ? -101 :
           wdl == WDLLoss        ? -1   : 0;
}

// Return the sign of a number (-1, 0, 1)
template <typename T> int sign_of(T val) {
    return (T(0) < val) - (val < T(0));
}

// Numbers in little endian used by sparseIndex[] to point into blockLength[]
struct SparseEntry {
    char block[4];   // Number of block
    char offset[2];  // Offset within the block
};

static_assert(sizeof(SparseEntry) == 6, "SparseEntry must be 6 bytes");

typedef uint16_t Sym; // Huffman symbol

struct LR {
    enum Side { Left, Right, Value };

    uint8_t lr[3]; // The first 12 bits is the left-hand symbol, the second 12
                   // bits is the right-hand symbol. If symbol has length 1,
                   // then the first byte is the stored value.
    template<Side S>
    Sym get() {
        return S == Left  ? ((lr[1] & 0xF) << 8) | lr[0] :
               S == Right ?  (lr[2] << 4) | (lr[1] >> 4) :
               S == Value ?   lr[0] : (assert(false), Sym(-1));
    }
};

static_assert(sizeof(LR) == 3, "LR tree entry must be 3 bytes");

// Tablebases data layout is structured as following:
//
//  TBFile:   memory maps/unmaps the physical .rtbw and .rtbz files
//  TBTable:  one object for each file with corresponding indexing information
//  TBTables: has ownership of TBTable objects, keeping a list and a hash

// class TBFile memory maps/unmaps the single .rtbw and .rtbz files. Files are
// memory mapped for best performance. Files are mapped at first access: at init
// time only existence of the file is checked.
class TBFile : public std::ifstream {

    std::string fname;

public:
    // Look for and open the file among the Paths directories where the .rtbw
    // and .rtbz files can be found. Multiple directories are separated by ";"
    // on Windows and by ":" on Unix-based operating systems.
    //
    // Example:
    // C:\tb\wdl345;C:\tb\wdl6;D:\tb\dtz345;D:\tb\dtz6
    static std::string Paths;

    TBFile(const std::string& f) {

#ifndef _WIN32
        constexpr char SepChar = ':';
#else
        constexpr char SepChar = ';';
#endif
        std::stringstream ss(Paths);
        std::string path;

        while (std::getline(ss, path, SepChar)) {
            fname = path + "/" + f;
            std::ifstream::open(fname);
            if (is_open())
                return;
        }
    }

    // Memory map the file and check it. File should be already open and will be
    // closed after mapping.
    uint8_t* map(void** baseAddress, uint64_t* mapping, TBType type) {

        assert(is_open());

        close(); // Need to re-open to get native file descriptor

#ifndef _WIN32
        struct stat statbuf;
        int fd = ::open(fname.c_str(), O_RDONLY);

        if (fd == -1)
            return *baseAddress = nullptr, nullptr;

        fstat(fd, &statbuf);
        *mapping = statbuf.st_size;
        *baseAddress = mmap(nullptr, statbuf.st_size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);

        if (*baseAddress == MAP_FAILED) {
            std::cerr << "Could not mmap() " << fname << std::endl;
            exit(1);
        }
#else
        HANDLE fd = CreateFile(fname.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

        if (fd == INVALID_HANDLE_VALUE)
            return *baseAddress = nullptr, nullptr;

        DWORD size_high;
        DWORD size_low = GetFileSize(fd, &size_high);
        HANDLE mmap = CreateFileMapping(fd, nullptr, PAGE_READONLY, size_high, size_low, nullptr);
        CloseHandle(fd);

        if (!mmap) {
            std::cerr << "CreateFileMapping() failed" << std::endl;
            exit(1);
        }

        *mapping = (uint64_t)mmap;
        *baseAddress = MapViewOfFile(mmap, FILE_MAP_READ, 0, 0, 0);

        if (!*baseAddress) {
            std::cerr << "MapViewOfFile() failed, name = " << fname
                      << ", error = " << GetLastError() << std::endl;
            exit(1);
        }
#endif
        uint8_t* data = (uint8_t*)*baseAddress;

        constexpr uint8_t Magics[][4] = { { 0xD7, 0x66, 0x0C, 0xA5 },
                                          { 0x71, 0xE8, 0x23, 0x5D } };

        if (memcmp(data, Magics[type == WDL], 4)) {
            std::cerr << "Corrupted table in file " << fname << std::endl;
            unmap(*baseAddress, *mapping);
            return *baseAddress = nullptr, nullptr;
        }

        return data + 4; // Skip Magics's header
    }

    static void unmap(void* baseAddress, uint64_t mapping) {

#ifndef _WIN32
        munmap(baseAddress, mapping);
#else
        UnmapViewOfFile(baseAddress);
        CloseHandle((HANDLE)mapping);
#endif
    }
};

std::string TBFile::Paths;

// struct PairsData contains low level indexing information to access TB data.
// There are 8, 4 or 2 PairsData records for each TBTable, according to type of
// table and if positions have pawns or not. It is populated at first access.
struct PairsData {
    uint8_t flags;                 // Table flags, see enum TBFlag
    uint8_t maxSymLen;             // Maximum length in bits of the Huffman symbols
    uint8_t minSymLen;             // Minimum length in bits of the Huffman symbols
    uint32_t blocksNum;            // Number of blocks in the TB file
    size_t sizeofBlock;            // Block size in bytes
    size_t span;                   // About every span values there is a SparseIndex[] entry
    Sym* lowestSym;                // lowestSym[l] is the symbol of length l with the lowest value
    LR* btree;                     // btree[sym] stores the left and right symbols that expand sym
    uint16_t* blockLength;         // Number of stored positions (minus one) for each block: 1..65536
    uint32_t blockLengthSize;      // Size of blockLength[] table: padded so it's bigger than blocksNum
    SparseEntry* sparseIndex;      // Partial indices into blockLength[]
    size_t sparseIndexSize;        // Size of SparseIndex[] table
    uint8_t* data;                 // Start of Huffman compressed data
    std::vector<uint64_t> base64;  // base64[l - min_sym_len] is the 64bit-padded lowest symbol of length l
    std::vector<uint8_t> symlen;   // Number of values (-1) represented by a given Huffman symbol: 1..256
    Piece pieces[TBPIECES];        // Position pieces: the order of pieces defines the groups
    uint64_t groupIdx[TBPIECES+1]; // Start index used for the encoding of the group's pieces
    int groupLen[TBPIECES+1];      // Number of pieces in a given group: KRKN -> (3, 1)
    uint16_t map_idx[4];           // WDLWin, WDLLoss, WDLCursedWin, WDLBlessedLoss (used in DTZ)
};

// struct TBTable contains indexing information to access the corresponding TBFile.
// There are 2 types of TBTable, corresponding to a WDL or a DTZ file. TBTable
// is populated at init time but the nested PairsData records are populated at
// first access, when the corresponding file is memory mapped.
template<TBType Type>
struct TBTable {
    typedef typename std::conditional<Type == WDL, WDLScore, int>::type Ret;

    static constexpr int Sides = Type == WDL ? 2 : 1;

    std::atomic_bool ready;
    void* baseAddress;
    uint8_t* map;
    uint64_t mapping;
    Key key;
    Key key2;
    int pieceCount;
    bool hasPawns;
    bool hasUniquePieces;
    uint8_t pawnCount[2]; // [Lead color / other color]
    PairsData items[Sides][4]; // [wtm / btm][FILE_A..FILE_D or 0]

    PairsData* get(int stm, int f) {
        return &items[stm % Sides][hasPawns ? f : 0];
    }

    TBTable() : ready(false), baseAddress(nullptr) {}
    explicit TBTable(const std::string& code);
    explicit TBTable(const TBTable<WDL>& wdl);

    ~TBTable() {
        if (baseAddress)
            TBFile::unmap(baseAddress, mapping);
    }
};

template<>
TBTable<WDL>::TBTable(const std::string& code) : TBTable() {

    StateInfo st;
    Position pos;

    key = pos.set(code, WHITE, &st).material_key();
    pieceCount = pos.count<ALL_PIECES>();
    hasPawns = pos.pieces(PAWN);

    hasUniquePieces = false;
    for (Color c = WHITE; c <= BLACK; ++c)
        for (PieceType pt = PAWN; pt < KING; ++pt)
            if (popcount(pos.pieces(c, pt)) == 1)
                hasUniquePieces = true;

    // Set the leading color. In case both sides have pawns the leading color
    // is the side with less pawns because this leads to better compression.
    bool c =   !pos.count<PAWN>(BLACK)
            || (   pos.count<PAWN>(WHITE)
                && pos.count<PAWN>(BLACK) >= pos.count<PAWN>(WHITE));

    pawnCount[0] = pos.count<PAWN>(c ? WHITE : BLACK);
    pawnCount[1] = pos.count<PAWN>(c ? BLACK : WHITE);

    key2 = pos.set(code, BLACK, &st).material_key();
}

template<>
TBTable<DTZ>::TBTable(const TBTable<WDL>& wdl) : TBTable() {

    // Use the corresponding WDL table to avoid recalculating all from scratch
    key = wdl.key;
    key2 = wdl.key2;
    pieceCount = wdl.pieceCount;
    hasPawns = wdl.hasPawns;
    hasUniquePieces = wdl.hasUniquePieces;
    pawnCount[0] = wdl.pawnCount[0];
    pawnCount[1] = wdl.pawnCount[1];
}

// class TBTables creates and keeps ownership of the TBTable objects, one for
// each TB file found. It supports a fast, hash based, table lookup. Populated
// at init time, accessed at probe time.
class TBTables {

    typedef std::tuple<Key, TBTable<WDL>*, TBTable<DTZ>*> Entry;

    static const int Size = 1 << 12; // 4K table, indexed by key's 12 lsb

    Entry hashTable[Size];

    std::deque<TBTable<WDL>> wdlTable;
    std::deque<TBTable<DTZ>> dtzTable;

    void insert(Key key, TBTable<WDL>* wdl, TBTable<DTZ>* dtz) {
        Entry* entry = &hashTable[(uint32_t)key & (Size - 1)];

        // Ensure last element is empty to avoid overflow when looking up
        for ( ; entry - hashTable < Size - 1; ++entry)
            if (std::get<KEY>(*entry) == key || !std::get<WDL>(*entry)) {
                *entry = std::make_tuple(key, wdl, dtz);
                return;
            }
        std::cerr << "TB hash table size too low!" << std::endl;
        exit(1);
    }

public:
    template<TBType Type>
    TBTable<Type>* get(Key key) {
        for (const Entry* entry = &hashTable[(uint32_t)key & (Size - 1)]; ; ++entry) {
            if (std::get<KEY>(*entry) == key || !std::get<Type>(*entry))
                return std::get<Type>(*entry);
        }
    }

    void clear() {
        memset(hashTable, 0, sizeof(hashTable));
        wdlTable.clear();
        dtzTable.clear();
    }
    size_t size() const { return wdlTable.size(); }
    void add(const std::vector<PieceType>& pieces);
};

TBTables TBTables;

// If the corresponding file exists two new objects TBTable<WDL> and TBTable<DTZ>
// are created and added to the lists and hash table. Called at init time.
void TBTables::add(const std::vector<PieceType>& pieces) {

    std::string code;

    for (PieceType pt : pieces)
        code += PieceToChar[pt];

    TBFile file(code.insert(code.find('K', 1), "v") + ".rtbw"); // KRK -> KRvK

    if (!file.is_open()) // Only WDL file is checked
        return;

    file.close();

    MaxCardinality = std::max((int)pieces.size(), MaxCardinality);

    wdlTable.emplace_back(code);
    dtzTable.emplace_back(wdlTable.back());

    // Insert into the hash keys for both colors: KRvK with KR white and black
    insert(wdlTable.back().key , &wdlTable.back(), &dtzTable.back());
    insert(wdlTable.back().key2, &wdlTable.back(), &dtzTable.back());
}

// TB tables are compressed with canonical Huffman code. The compressed data is divided into
// blocks of size d->sizeofBlock, and each block stores a variable number of symbols.
// Each symbol represents either a WDL or a (remapped) DTZ value, or a pair of other symbols
// (recursively). If you keep expanding the symbols in a block, you end up with up to 65536
// WDL or DTZ values. Each symbol represents up to 256 values and will correspond after
// Huffman coding to at least 1 bit. So a block of 32 bytes corresponds to at most
// 32 x 8 x 256 = 65536 values. This maximum is only reached for tables that consist mostly
// of draws or mostly of wins, but such tables are actually quite common. In principle, the
// blocks in WDL tables are 64 bytes long (and will be aligned on cache lines). But for
// mostly-draw or mostly-win tables this can leave many 64-byte blocks only half-filled, so
// in such cases blocks are 32 bytes long. The blocks of DTZ tables are up to 1024 bytes long.
// The generator picks the size that leads to the smallest table. The "book" of symbols and
// Huffman codes is the same for all blocks in the table. A non-symmetric pawnless TB file
// will have one table for wtm and one for btm, a TB file with pawns will have tables per
// file a,b,c,d also in this case one set for wtm and one for btm.
int decompress_pairs(PairsData* d, uint64_t idx) {

    // Special case where all table positions store the same value
    if (d->flags & TBFlag::SingleValue)
        return d->minSymLen;

    // First we need to locate the right block that stores the value at index "idx".
    // Because each block n stores blockLength[n] + 1 values, the index i of the block
    // that contains the value at position idx is:
    //
    //                    for (i = -1, sum = 0; sum <= idx; i++)
    //                        sum += blockLength[i + 1] + 1;
    //
    // This can be slow, so we use SparseIndex[] populated with a set of SparseEntry that
    // point to known indices into blockLength[]. Namely SparseIndex[k] is a SparseEntry
    // that stores the blockLength[] index and the offset within that block of the value
    // with index I(k), where:
    //
    //       I(k) = k * d->span + d->span / 2      (1)

    // First step is to get the 'k' of the I(k) nearest to our idx, using definition (1)
    uint32_t k = idx / d->span;

    // Then we read the corresponding SparseIndex[] entry
    uint32_t block = number<uint32_t, LittleEndian>(&d->sparseIndex[k].block);
    int offset     = number<uint16_t, LittleEndian>(&d->sparseIndex[k].offset);

    // Now compute the difference idx - I(k). From definition of k we know that
    //
    //       idx = k * d->span + idx % d->span    (2)
    //
    // So from (1) and (2) we can compute idx - I(K):
    int diff = idx % d->span - d->span / 2;

    // Sum the above to offset to find the offset corresponding to our idx
    offset += diff;

    // Move to previous/next block, until we reach the correct block that contains idx,
    // that is when 0 <= offset <= d->blockLength[block]
    while (offset < 0)
        offset += d->blockLength[--block] + 1;

    while (offset > d->blockLength[block])
        offset -= d->blockLength[block++] + 1;

    // Finally, we find the start address of our block of canonical Huffman symbols
    uint32_t* ptr = (uint32_t*)(d->data + block * d->sizeofBlock);

    // Read the first 64 bits in our block, this is a (truncated) sequence of
    // unknown number of symbols of unknown length but we know the first one
    // is at the beginning of this 64 bits sequence.
    uint64_t buf64 = number<uint64_t, BigEndian>(ptr); ptr += 2;
    int buf64Size = 64;
    Sym sym;

    while (true) {
        int len = 0; // This is the symbol length - d->min_sym_len

        // Now get the symbol length. For any symbol s64 of length l right-padded
        // to 64 bits we know that d->base64[l-1] >= s64 >= d->base64[l] so we
        // can find the symbol length iterating through base64[].
        while (buf64 < d->base64[len])
            ++len;

        // All the symbols of a given length are consecutive integers (numerical
        // sequence property), so we can compute the offset of our symbol of
        // length len, stored at the beginning of buf64.
        sym = (buf64 - d->base64[len]) >> (64 - len - d->minSymLen);

        // Now add the value of the lowest symbol of length len to get our symbol
        sym += number<Sym, LittleEndian>(&d->lowestSym[len]);

        // If our offset is within the number of values represented by symbol sym
        // we are done...
        if (offset < d->symlen[sym] + 1)
            break;

        // ...otherwise update the offset and continue to iterate
        offset -= d->symlen[sym] + 1;
        len += d->minSymLen; // Get the real length
        buf64 <<= len;       // Consume the just processed symbol
        buf64Size -= len;

        if (buf64Size <= 32) { // Refill the buffer
            buf64Size += 32;
            buf64 |= (uint64_t)number<uint32_t, BigEndian>(ptr++) << (64 - buf64Size);
        }
    }

    // Ok, now we have our symbol that expands into d->symlen[sym] + 1 symbols.
    // We binary-search for our value recursively expanding into the left and
    // right child symbols until we reach a leaf node where symlen[sym] + 1 == 1
    // that will store the value we need.
    while (d->symlen[sym]) {

        Sym left = d->btree[sym].get<LR::Left>();

        // If a symbol contains 36 sub-symbols (d->symlen[sym] + 1 = 36) and
        // expands in a pair (d->symlen[left] = 23, d->symlen[right] = 11), then
        // we know that, for instance the ten-th value (offset = 10) will be on
        // the left side because in Recursive Pairing child symbols are adjacent.
        if (offset < d->symlen[left] + 1)
            sym = left;
        else {
            offset -= d->symlen[left] + 1;
            sym = d->btree[sym].get<LR::Right>();
        }
    }

    return d->btree[sym].get<LR::Value>();
}

bool check_dtz_stm(TBTable<WDL>*, int, File) { return true; }

bool check_dtz_stm(TBTable<DTZ>* entry, int stm, File f) {

    auto flags = entry->get(stm, f)->flags;
    return   (flags & TBFlag::STM) == stm
          || ((entry->key == entry->key2) && !entry->hasPawns);
}

// DTZ scores are sorted by frequency of occurrence and then assigned the
// values 0, 1, 2, ... in order of decreasing frequency. This is done for each
// of the four WDLScore values. The mapping information necessary to reconstruct
// the original values is stored in the TB file and read during map[] init.
WDLScore map_score(TBTable<WDL>*, File, int value, WDLScore) { return WDLScore(value - 2); }

int map_score(TBTable<DTZ>* entry, File f, int value, WDLScore wdl) {

    constexpr int WDLMap[] = { 1, 3, 0, 2, 0 };

    auto flags = entry->get(0, f)->flags;

    uint8_t* map = entry->map;
    uint16_t* idx = entry->get(0, f)->map_idx;
    if (flags & TBFlag::Mapped)
        value = map[idx[WDLMap[wdl + 2]] + value];

    // DTZ tables store distance to zero in number of moves or plies. We
    // want to return plies, so we have convert to plies when needed.
    if (   (wdl == WDLWin  && !(flags & TBFlag::WinPlies))
        || (wdl == WDLLoss && !(flags & TBFlag::LossPlies))
        ||  wdl == WDLCursedWin
        ||  wdl == WDLBlessedLoss)
        value *= 2;

    return value + 1;
}

// Compute a unique index out of a position and use it to probe the TB file. To
// encode k pieces of same type and color, first sort the pieces by square in
// ascending order s1 <= s2 <= ... <= sk then compute the unique index as:
//
//      idx = Binomial[1][s1] + Binomial[2][s2] + ... + Binomial[k][sk]
//
template<typename T, typename Ret = typename T::Ret>
Ret do_probe_table(const Position& pos, T* entry, WDLScore wdl, ProbeState* result) {

    Square squares[TBPIECES];
    Piece pieces[TBPIECES];
    uint64_t idx;
    int next = 0, size = 0, leadPawnsCnt = 0;
    PairsData* d;
    Bitboard b, leadPawns = 0;
    File tbFile = FILE_A;

    // A given TB entry like KRK has associated two material keys: KRvk and Kvkr.
    // If both sides have the same pieces keys are equal. In this case TB tables
    // only store the 'white to move' case, so if the position to lookup has black
    // to move, we need to switch the color and flip the squares before to lookup.
    bool symmetricBlackToMove = (entry->key == entry->key2 && pos.side_to_move());

    // TB files are calculated for white as stronger side. For instance we have
    // KRvK, not KvKR. A position where stronger side is white will have its
    // material key == entry->key, otherwise we have to switch the color and
    // flip the squares before to lookup.
    bool blackStronger = (pos.material_key() != entry->key);

    int flipColor   = (symmetricBlackToMove || blackStronger) * 8;
    int flipSquares = (symmetricBlackToMove || blackStronger) * 070;
    int stm         = (symmetricBlackToMove || blackStronger) ^ pos.side_to_move();

    // For pawns, TB files store 4 separate tables according if leading pawn is on
    // file a, b, c or d after reordering. The leading pawn is the one with maximum
    // MapPawns[] value, that is the one most toward the edges and with lowest rank.
    if (entry->hasPawns) {

        // In all the 4 tables, pawns are at the beginning of the piece sequence and
        // their color is the reference one. So we just pick the first one.
        Piece pc = Piece(entry->get(0, 0)->pieces[0] ^ flipColor);

        assert(type_of(pc) == PAWN);

        leadPawns = b = pos.pieces(color_of(pc), PAWN);
        do
            squares[size++] = pop_lsb(&b) ^ flipSquares;
        while (b);

        leadPawnsCnt = size;

        std::swap(squares[0], *std::max_element(squares, squares + leadPawnsCnt, pawns_comp));

        tbFile = file_of(squares[0]);
        if (tbFile > FILE_D)
            tbFile = file_of(squares[0] ^ 7); // Horizontal flip: SQ_H1 -> SQ_A1
    }

    // DTZ tables are one-sided, i.e. they store positions only for white to
    // move or only for black to move, so check for side to move to be stm,
    // early exit otherwise.
    if (!check_dtz_stm(entry, stm, tbFile))
        return *result = CHANGE_STM, Ret();

    // Now we are ready to get all the position pieces (but the lead pawns) and
    // directly map them to the correct color and square.
    b = pos.pieces() ^ leadPawns;
    do {
        Square s = pop_lsb(&b);
        squares[size] = s ^ flipSquares;
        pieces[size++] = Piece(pos.piece_on(s) ^ flipColor);
    } while (b);

    assert(size >= 2);

    d = entry->get(stm, tbFile);

    // Then we reorder the pieces to have the same sequence as the one stored
    // in pieces[i]: the sequence that ensures the best compression.
    for (int i = leadPawnsCnt; i < size; ++i)
        for (int j = i; j < size; ++j)
            if (d->pieces[i] == pieces[j])
            {
                std::swap(pieces[i], pieces[j]);
                std::swap(squares[i], squares[j]);
                break;
            }

    // Now we map again the squares so that the square of the lead piece is in
    // the triangle A1-D1-D4.
    if (file_of(squares[0]) > FILE_D)
        for (int i = 0; i < size; ++i)
            squares[i] ^= 7; // Horizontal flip: SQ_H1 -> SQ_A1

    // Encode leading pawns starting with the one with minimum MapPawns[] and
    // proceeding in ascending order.
    if (entry->hasPawns) {
        idx = LeadPawnIdx[leadPawnsCnt][squares[0]];

        std::sort(squares + 1, squares + leadPawnsCnt, pawns_comp);

        for (int i = 1; i < leadPawnsCnt; ++i)
            idx += Binomial[i][MapPawns[squares[i]]];

        goto encode_remaining; // With pawns we have finished special treatments
    }

    // In positions withouth pawns, we further flip the squares to ensure leading
    // piece is below RANK_5.
    if (rank_of(squares[0]) > RANK_4)
        for (int i = 0; i < size; ++i)
            squares[i] ^= 070; // Vertical flip: SQ_A8 -> SQ_A1

    // Look for the first piece of the leading group not on the A1-D4 diagonal
    // and ensure it is mapped below the diagonal.
    for (int i = 0; i < d->groupLen[0]; ++i) {
        if (!off_A1H8(squares[i]))
            continue;

        if (off_A1H8(squares[i]) > 0) // A1-H8 diagonal flip: SQ_A3 -> SQ_C3
            for (int j = i; j < size; ++j)
                squares[j] = Square(((squares[j] >> 3) | (squares[j] << 3)) & 63);
        break;
    }

    // Encode the leading group.
    //
    // Suppose we have KRvK. Let's say the pieces are on square numbers wK, wR
    // and bK (each 0...63). The simplest way to map this position to an index
    // is like this:
    //
    //   index = wK * 64 * 64 + wR * 64 + bK;
    //
    // But this way the TB is going to have 64*64*64 = 262144 positions, with
    // lots of positions being equivalent (because they are mirrors of each
    // other) and lots of positions being invalid (two pieces on one square,
    // adjacent kings, etc.).
    // Usually the first step is to take the wK and bK together. There are just
    // 462 ways legal and not-mirrored ways to place the wK and bK on the board.
    // Once we have placed the wK and bK, there are 62 squares left for the wR
    // Mapping its square from 0..63 to available squares 0..61 can be done like:
    //
    //   wR -= (wR > wK) + (wR > bK);
    //
    // In words: if wR "comes later" than wK, we deduct 1, and the same if wR
    // "comes later" than bK. In case of two same pieces like KRRvK we want to
    // place the two Rs "together". If we have 62 squares left, we can place two
    // Rs "together" in 62 * 61 / 2 ways (we divide by 2 because rooks can be
    // swapped and still get the same position.)
    //
    // In case we have at least 3 unique pieces (inlcuded kings) we encode them
    // together.
    if (entry->hasUniquePieces) {

        int adjust1 =  squares[1] > squares[0];
        int adjust2 = (squares[2] > squares[0]) + (squares[2] > squares[1]);

        // First piece is below a1-h8 diagonal. MapA1D1D4[] maps the b1-d1-d3
        // triangle to 0...5. There are 63 squares for second piece and and 62
        // (mapped to 0...61) for the third.
        if (off_A1H8(squares[0]))
            idx = (   MapA1D1D4[squares[0]]  * 63
                   + (squares[1] - adjust1)) * 62
                   +  squares[2] - adjust2;

        // First piece is on a1-h8 diagonal, second below: map this occurence to
        // 6 to differentiate from the above case, rank_of() maps a1-d4 diagonal
        // to 0...3 and finally MapB1H1H7[] maps the b1-h1-h7 triangle to 0..27.
        else if (off_A1H8(squares[1]))
            idx = (  6 * 63 + rank_of(squares[0]) * 28
                   + MapB1H1H7[squares[1]])       * 62
                   + squares[2] - adjust2;

        // First two pieces are on a1-h8 diagonal, third below
        else if (off_A1H8(squares[2]))
            idx =  6 * 63 * 62 + 4 * 28 * 62
                 +  rank_of(squares[0])        * 7 * 28
                 + (rank_of(squares[1]) - adjust1) * 28
                 +  MapB1H1H7[squares[2]];

        // All 3 pieces on the diagonal a1-h8
        else
            idx = 6 * 63 * 62 + 4 * 28 * 62 + 4 * 7 * 28
                 +  rank_of(squares[0])         * 7 * 6
                 + (rank_of(squares[1]) - adjust1)  * 6
                 + (rank_of(squares[2]) - adjust2);
    } else
        // We don't have at least 3 unique pieces, like in KRRvKBB, just map
        // the kings.
        idx = MapKK[MapA1D1D4[squares[0]]][squares[1]];

encode_remaining:
    idx *= d->groupIdx[0];
    Square* groupSq = squares + d->groupLen[0];

    // Encode remainig pawns then pieces according to square, in ascending order
    bool remainingPawns = entry->hasPawns && entry->pawnCount[1];

    while (d->groupLen[++next])
    {
        std::sort(groupSq, groupSq + d->groupLen[next]);
        uint64_t n = 0;

        // Map down a square if "comes later" than a square in the previous
        // groups (similar to what done earlier for leading group pieces).
        for (int i = 0; i < d->groupLen[next]; ++i)
        {
            auto f = [&](Square s) { return groupSq[i] > s; };
            auto adjust = std::count_if(squares, groupSq, f);
            n += Binomial[i + 1][groupSq[i] - adjust - 8 * remainingPawns];
        }

        remainingPawns = false;
        idx += n * d->groupIdx[next];
        groupSq += d->groupLen[next];
    }

    // Now that we have the index, decompress the pair and get the score
    return map_score(entry, tbFile, decompress_pairs(d, idx), wdl);
}

// Group together pieces that will be encoded together. The general rule is that
// a group contains pieces of same type and color. The exception is the leading
// group that, in case of positions withouth pawns, can be formed by 3 different
// pieces (default) or by the king pair when there is not a unique piece apart
// from the kings. When there are pawns, pawns are always first in pieces[].
//
// As example KRKN -> KRK + N, KNNK -> KK + NN, KPPKP -> P + PP + K + K
//
// The actual grouping depends on the TB generator and can be inferred from the
// sequence of pieces in piece[] array.
template<typename T>
void set_groups(T& e, PairsData* d, int order[], File f) {

    int n = 0, firstLen = e.hasPawns ? 0 : e.hasUniquePieces ? 3 : 2;
    d->groupLen[n] = 1;

    // Number of pieces per group is stored in groupLen[], for instance in KRKN
    // the encoder will default on '111', so groupLen[] will be (3, 1).
    for (int i = 1; i < e.pieceCount; ++i)
        if (--firstLen > 0 || d->pieces[i] == d->pieces[i - 1])
            d->groupLen[n]++;
        else
            d->groupLen[++n] = 1;

    d->groupLen[++n] = 0; // Zero-terminated

    // The sequence in pieces[] defines the groups, but not the order in which
    // they are encoded. If the pieces in a group g can be combined on the board
    // in N(g) different ways, then the position encoding will be of the form:
    //
    //           g1 * N(g2) * N(g3) + g2 * N(g3) + g3
    //
    // This ensures unique encoding for the whole position. The order of the
    // groups is a per-table parameter and could not follow the canonical leading
    // pawns/pieces -> remainig pawns -> remaining pieces. In particular the
    // first group is at order[0] position and the remaining pawns, when present,
    // are at order[1] position.
    bool pp = e.hasPawns && e.pawnCount[1]; // Pawns on both sides
    int next = pp ? 2 : 1;
    int freeSquares = 64 - d->groupLen[0] - (pp ? d->groupLen[1] : 0);
    uint64_t idx = 1;

    for (int k = 0; next < n || k == order[0] || k == order[1]; ++k)
        if (k == order[0]) // Leading pawns or pieces
        {
            d->groupIdx[0] = idx;
            idx *=         e.hasPawns ? LeadPawnsSize[d->groupLen[0]][f]
                  : e.hasUniquePieces ? 31332 : 462;
        }
        else if (k == order[1]) // Remaining pawns
        {
            d->groupIdx[1] = idx;
            idx *= Binomial[d->groupLen[1]][48 - d->groupLen[0]];
        }
        else // Remainig pieces
        {
            d->groupIdx[next] = idx;
            idx *= Binomial[d->groupLen[next]][freeSquares];
            freeSquares -= d->groupLen[next++];
        }

    d->groupIdx[n] = idx;
}

// In Recursive Pairing each symbol represents a pair of childern symbols. So
// read d->btree[] symbols data and expand each one in his left and right child
// symbol until reaching the leafs that represent the symbol value.
uint8_t set_symlen(PairsData* d, Sym s, std::vector<bool>& visited) {

    visited[s] = true; // We can set it now because tree is acyclic
    Sym sr = d->btree[s].get<LR::Right>();

    if (sr == 0xFFF)
        return 0;

    Sym sl = d->btree[s].get<LR::Left>();

    if (!visited[sl])
        d->symlen[sl] = set_symlen(d, sl, visited);

    if (!visited[sr])
        d->symlen[sr] = set_symlen(d, sr, visited);

    return d->symlen[sl] + d->symlen[sr] + 1;
}

uint8_t* set_sizes(PairsData* d, uint8_t* data) {

    d->flags = *data++;

    if (d->flags & TBFlag::SingleValue) {
        d->blocksNum = d->blockLengthSize = 0;
        d->span = d->sparseIndexSize = 0; // Broken MSVC zero-init
        d->minSymLen = *data++; // Here we store the single value
        return data;
    }

    // groupLen[] is a zero-terminated list of group lengths, the last groupIdx[]
    // element stores the biggest index that is the tb size.
    uint64_t tbSize = d->groupIdx[std::find(d->groupLen, d->groupLen + 7, 0) - d->groupLen];

    d->sizeofBlock = 1ULL << *data++;
    d->span = 1ULL << *data++;
    d->sparseIndexSize = (tbSize + d->span - 1) / d->span; // Round up
    auto padding = number<uint8_t, LittleEndian>(data++);
    d->blocksNum = number<uint32_t, LittleEndian>(data); data += sizeof(uint32_t);
    d->blockLengthSize = d->blocksNum + padding; // Padded to ensure SparseIndex[]
                                                 // does not point out of range.
    d->maxSymLen = *data++;
    d->minSymLen = *data++;
    d->lowestSym = (Sym*)data;
    d->base64.resize(d->maxSymLen - d->minSymLen + 1);

    // The canonical code is ordered such that longer symbols (in terms of
    // the number of bits of their Huffman code) have lower numeric value,
    // so that d->lowestSym[i] >= d->lowestSym[i+1] (when read as LittleEndian).
    // Starting from this we compute a base64[] table indexed by symbol length
    // and containing 64 bit values so that d->base64[i] >= d->base64[i+1].
    // See http://www.eecs.harvard.edu/~michaelm/E210/huffman.pdf
    for (int i = d->base64.size() - 2; i >= 0; --i) {
        d->base64[i] = (d->base64[i + 1] + number<Sym, LittleEndian>(&d->lowestSym[i])
                                         - number<Sym, LittleEndian>(&d->lowestSym[i + 1])) / 2;

        assert(d->base64[i] * 2 >= d->base64[i+1]);
    }

    // Now left-shift by an amount so that d->base64[i] gets shifted 1 bit more
    // than d->base64[i+1] and given the above assert condition, we ensure that
    // d->base64[i] >= d->base64[i+1]. Moreover for any symbol s64 of length i
    // and right-padded to 64 bits holds d->base64[i-1] >= s64 >= d->base64[i].
    for (size_t i = 0; i < d->base64.size(); ++i)
        d->base64[i] <<= 64 - i - d->minSymLen; // Right-padding to 64 bits

    data += d->base64.size() * sizeof(Sym);
    d->symlen.resize(number<uint16_t, LittleEndian>(data)); data += sizeof(uint16_t);
    d->btree = (LR*)data;

    // The comrpession scheme used is "Recursive Pairing", that replaces the most
    // frequent adjacent pair of symbols in the source message by a new symbol,
    // reevaluating the frequencies of all of the symbol pairs with respect to
    // the extended alphabet, and then repeating the process.
    // See http://www.larsson.dogma.net/dcc99.pdf
    std::vector<bool> visited(d->symlen.size());

    for (Sym sym = 0; sym < d->symlen.size(); ++sym)
        if (!visited[sym])
            d->symlen[sym] = set_symlen(d, sym, visited);

    return data + d->symlen.size() * sizeof(LR) + (d->symlen.size() & 1);
}

uint8_t* set_dtz_map(TBTable<WDL>&, uint8_t* data, File) { return data; }

uint8_t* set_dtz_map(TBTable<DTZ>& e, uint8_t* data, File maxFile) {

    e.map = data;

    for (File f = FILE_A; f <= maxFile; ++f) {
        if (e.get(0, f)->flags & TBFlag::Mapped)
            for (int i = 0; i < 4; ++i) { // Sequence like 3,x,x,x,1,x,0,2,x,x
                e.get(0, f)->map_idx[i] = (uint16_t)(data - e.map + 1);
                data += *data + 1;
            }
    }

    return data += (uintptr_t)data & 1; // Word alignment
}

// Populate entry's PairsData records with data from the just memory mapped file.
// Called at first access.
template<typename T>
void set(T& e, uint8_t* data) {

    PairsData* d;

    enum { Split = 1, HasPawns = 2 };

    assert(e.hasPawns        == !!(*data & HasPawns));
    assert((e.key != e.key2) == !!(*data & Split));

    data++; // First byte stores flags

    const int sides = T::Sides == 2 && (e.key != e.key2) ? 2 : 1;
    const File maxFile = e.hasPawns ? FILE_D : FILE_A;

    bool pp = e.hasPawns && e.pawnCount[1]; // Pawns on both sides

    assert(!pp || e.pawnCount[0]);

    for (File f = FILE_A; f <= maxFile; ++f) {

        for (int i = 0; i < sides; i++)
            *e.get(i, f) = PairsData();

        int order[][2] = { { *data & 0xF, pp ? *(data + 1) & 0xF : 0xF },
                           { *data >>  4, pp ? *(data + 1) >>  4 : 0xF } };
        data += 1 + pp;

        for (int k = 0; k < e.pieceCount; ++k, ++data)
            for (int i = 0; i < sides; i++)
                e.get(i, f)->pieces[k] = Piece(i ? *data >>  4 : *data & 0xF);

        for (int i = 0; i < sides; ++i)
            set_groups(e, e.get(i, f), order[i], f);
    }

    data += (uintptr_t)data & 1; // Word alignment

    for (File f = FILE_A; f <= maxFile; ++f)
        for (int i = 0; i < sides; i++)
            data = set_sizes(e.get(i, f), data);

    data = set_dtz_map(e, data, maxFile);

    for (File f = FILE_A; f <= maxFile; ++f)
        for (int i = 0; i < sides; i++) {
            (d = e.get(i, f))->sparseIndex = (SparseEntry*)data;
            data += d->sparseIndexSize * sizeof(SparseEntry);
        }

    for (File f = FILE_A; f <= maxFile; ++f)
        for (int i = 0; i < sides; i++) {
            (d = e.get(i, f))->blockLength = (uint16_t*)data;
            data += d->blockLengthSize * sizeof(uint16_t);
        }

    for (File f = FILE_A; f <= maxFile; ++f)
        for (int i = 0; i < sides; i++) {
            data = (uint8_t*)(((uintptr_t)data + 0x3F) & ~0x3F); // 64 byte alignment
            (d = e.get(i, f))->data = data;
            data += d->blocksNum * d->sizeofBlock;
        }
}

// If the TB file corresponding to the given position is already memory mapped
// then return its base address, otherwise try to memory map and init it. Called
// at every probe, memory map and init only at first access. Function is thread
// safe and can be called concurrently.
template<TBType Type>
void* mapped(TBTable<Type>& e, const Position& pos) {

    static Mutex mutex;

    // Use 'aquire' to avoid a thread reads 'ready' == true while another is
    // still working, this could happen due to compiler reordering.
    if (e.ready.load(std::memory_order_acquire))
        return e.baseAddress; // Could be nullptr if file does not exsist

    std::unique_lock<Mutex> lk(mutex);

    if (e.ready.load(std::memory_order_relaxed)) // Recheck under lock
        return e.baseAddress;

    // Pieces strings in decreasing order for each color, like ("KPP","KR")
    std::string fname, w, b;
    for (PieceType pt = KING; pt >= PAWN; --pt) {
        w += std::string(popcount(pos.pieces(WHITE, pt)), PieceToChar[pt]);
        b += std::string(popcount(pos.pieces(BLACK, pt)), PieceToChar[pt]);
    }

    fname =  (e.key == pos.material_key() ? w + 'v' + b : b + 'v' + w)
           + (Type == WDL ? ".rtbw" : ".rtbz");

    uint8_t* data = TBFile(fname).map(&e.baseAddress, &e.mapping, Type);

    if (data)
        set(e, data);

    e.ready.store(true, std::memory_order_release);
    return e.baseAddress;
}

template<TBType Type, typename Ret = typename TBTable<Type>::Ret>
Ret probe_table(const Position& pos, ProbeState* result, WDLScore wdl = WDLDraw) {

    if (pos.count<ALL_PIECES>() == 2) // KvK
        return Ret(WDLDraw);

    TBTable<Type>* entry = TBTables.get<Type>(pos.material_key());

    if (!entry || !mapped(*entry, pos))
        return *result = FAIL, Ret();

    return do_probe_table(pos, entry, wdl, result);
}

// For a position where the side to move has a winning capture it is not necessary
// to store a winning value so the generator treats such positions as "don't cares"
// and tries to assign to it a value that improves the compression ratio. Similarly,
// if the side to move has a drawing capture, then the position is at least drawn.
// If the position is won, then the TB needs to store a win value. But if the
// position is drawn, the TB may store a loss value if that is better for compression.
// All of this means that during probing, the engine must look at captures and probe
// their results and must probe the position itself. The "best" result of these
// probes is the correct result for the position.
// DTZ table don't store values when a following move is a zeroing winning move
// (winning capture or winning pawn move). Also DTZ store wrong values for positions
// where the best move is an ep-move (even if losing). So in all these cases set
// the state to ZEROING_BEST_MOVE.
template<bool CheckZeroingMoves = false>
WDLScore search(Position& pos, ProbeState* result) {

    WDLScore value, bestValue = WDLLoss;
    StateInfo st;

    auto moveList = MoveList<LEGAL>(pos);
    size_t totalCount = moveList.size(), moveCount = 0;

    for (const Move& move : moveList)
    {
        if (   !pos.capture(move)
            && (!CheckZeroingMoves || type_of(pos.moved_piece(move)) != PAWN))
            continue;

        moveCount++;

        pos.do_move(move, st);
        value = -search(pos, result);
        pos.undo_move(move);

        if (*result == FAIL)
            return WDLDraw;

        if (value > bestValue)
        {
            bestValue = value;

            if (value >= WDLWin)
            {
                *result = ZEROING_BEST_MOVE; // Winning DTZ-zeroing move
                return value;
            }
        }
    }

    // In case we have already searched all the legal moves we don't have to probe
    // the TB because the stored score could be wrong. For instance TB tables
    // do not contain information on position with ep rights, so in this case
    // the result of probe_wdl_table is wrong. Also in case of only capture
    // moves, for instance here 4K3/4q3/6p1/2k5/6p1/8/8/8 w - - 0 7, we have to
    // return with ZEROING_BEST_MOVE set.
    bool noMoreMoves = (moveCount && moveCount == totalCount);

    if (noMoreMoves)
        value = bestValue;
    else
    {
        value = probe_table<WDL>(pos, result);

        if (*result == FAIL)
            return WDLDraw;
    }

    // DTZ stores a "don't care" value if bestValue is a win
    if (bestValue >= value)
        return *result = (   bestValue > WDLDraw
                          || noMoreMoves ? ZEROING_BEST_MOVE : OK), bestValue;

    return *result = OK, value;
}

} // namespace


/// Tablebases::init() is called at startup and after every change to
/// "SyzygyPath" UCI option to (re)create the various tables. It is not thread
/// safe, nor it needs to be.
void Tablebases::init(const std::string& paths) {

    TBTables.clear();
    MaxCardinality = 0;
    TBFile::Paths = paths;

    if (paths.empty() || paths == "<empty>")
        return;

    // MapB1H1H7[] encodes a square below a1-h8 diagonal to 0..27
    int code = 0;
    for (Square s = SQ_A1; s <= SQ_H8; ++s)
        if (off_A1H8(s) < 0)
            MapB1H1H7[s] = code++;

    // MapA1D1D4[] encodes a square in the a1-d1-d4 triangle to 0..9
    std::vector<Square> diagonal;
    code = 0;
    for (Square s = SQ_A1; s <= SQ_D4; ++s)
        if (off_A1H8(s) < 0 && file_of(s) <= FILE_D)
            MapA1D1D4[s] = code++;

        else if (!off_A1H8(s) && file_of(s) <= FILE_D)
            diagonal.push_back(s);

    // Diagonal squares are encoded as last ones
    for (auto s : diagonal)
        MapA1D1D4[s] = code++;

    // MapKK[] encodes all the 461 possible legal positions of two kings where
    // the first is in the a1-d1-d4 triangle. If the first king is on the a1-d4
    // diagonal, the other one shall not to be above the a1-h8 diagonal.
    std::vector<std::pair<int, Square>> bothOnDiagonal;
    code = 0;
    for (int idx = 0; idx < 10; idx++)
        for (Square s1 = SQ_A1; s1 <= SQ_D4; ++s1)
            if (MapA1D1D4[s1] == idx && (idx || s1 == SQ_B1)) // SQ_B1 is mapped to 0
            {
                for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
                    if ((PseudoAttacks[KING][s1] | s1) & s2)
                        continue; // Illegal position

                    else if (!off_A1H8(s1) && off_A1H8(s2) > 0)
                        continue; // First on diagonal, second above

                    else if (!off_A1H8(s1) && !off_A1H8(s2))
                        bothOnDiagonal.push_back(std::make_pair(idx, s2));

                    else
                        MapKK[idx][s2] = code++;
            }

    // Legal positions with both kings on diagonal are encoded as last ones
    for (auto p : bothOnDiagonal)
        MapKK[p.first][p.second] = code++;

    // Binomial[] stores the Binomial Coefficents using Pascal rule. There
    // are Binomial[k][n] ways to choose k elements from a set of n elements.
    Binomial[0][0] = 1;

    for (int n = 1; n < 64; n++) // Squares
        for (int k = 0; k < 6 && k <= n; ++k) // Pieces
            Binomial[k][n] =  (k > 0 ? Binomial[k - 1][n - 1] : 0)
                            + (k < n ? Binomial[k    ][n - 1] : 0);

    // MapPawns[s] encodes squares a2-h7 to 0..47. This is the number of possible
    // available squares when the leading one is in 's'. Moreover the pawn with
    // highest MapPawns[] is the leading pawn, the one nearest the edge and,
    // among pawns with same file, the one with lowest rank.
    int availableSquares = 47; // Available squares when lead pawn is in a2

    // Init the tables for the encoding of leading pawns group: with 6-men TB we
    // can have up to 4 leading pawns (KPPPPK).
    for (int leadPawnsCnt = 1; leadPawnsCnt <= 4; ++leadPawnsCnt)
        for (File f = FILE_A; f <= FILE_D; ++f)
        {
            // Restart the index at every file because TB table is splitted
            // by file, so we can reuse the same index for different files.
            int idx = 0;

            // Sum all possible combinations for a given file, starting with
            // the leading pawn on rank 2 and increasing the rank.
            for (Rank r = RANK_2; r <= RANK_7; ++r)
            {
                Square sq = make_square(f, r);

                // Compute MapPawns[] at first pass.
                // If sq is the leading pawn square, any other pawn cannot be
                // below or more toward the edge of sq. There are 47 available
                // squares when sq = a2 and reduced by 2 for any rank increase
                // due to mirroring: sq == a3 -> no a2, h2, so MapPawns[a3] = 45
                if (leadPawnsCnt == 1)
                {
                    MapPawns[sq] = availableSquares--;
                    MapPawns[sq ^ 7] = availableSquares--; // Horizontal flip
                }
                LeadPawnIdx[leadPawnsCnt][sq] = idx;
                idx += Binomial[leadPawnsCnt - 1][MapPawns[sq]];
            }
            // After a file is traversed, store the cumulated per-file index
            LeadPawnsSize[leadPawnsCnt][f] = idx;
        }

    // Add entries in TB tables if the corresponding ".rtbw" file exsists
    for (PieceType p1 = PAWN; p1 < KING; ++p1) {
        TBTables.add({KING, p1, KING});

        for (PieceType p2 = PAWN; p2 <= p1; ++p2) {
            TBTables.add({KING, p1, p2, KING});
            TBTables.add({KING, p1, KING, p2});

            for (PieceType p3 = PAWN; p3 < KING; ++p3)
                TBTables.add({KING, p1, p2, KING, p3});

            for (PieceType p3 = PAWN; p3 <= p2; ++p3) {
                TBTables.add({KING, p1, p2, p3, KING});

                for (PieceType p4 = PAWN; p4 <= p3; ++p4)
                    TBTables.add({KING, p1, p2, p3, p4, KING});

                for (PieceType p4 = PAWN; p4 < KING; ++p4)
                    TBTables.add({KING, p1, p2, p3, KING, p4});
            }

            for (PieceType p3 = PAWN; p3 <= p1; ++p3)
                for (PieceType p4 = PAWN; p4 <= (p1 == p3 ? p2 : p3); ++p4)
                    TBTables.add({KING, p1, p2, KING, p3, p4});
        }
    }

    sync_cout << "info string Found " << TBTables.size() << " tablebases" << sync_endl;
}

// Probe the WDL table for a particular position.
// If *result != FAIL, the probe was successful.
// The return value is from the point of view of the side to move:
// -2 : loss
// -1 : loss, but draw under 50-move rule
//  0 : draw
//  1 : win, but draw under 50-move rule
//  2 : win
WDLScore Tablebases::probe_wdl(Position& pos, ProbeState* result) {

    *result = OK;
    return search(pos, result);
}

// Probe the DTZ table for a particular position.
// If *result != FAIL, the probe was successful.
// The return value is from the point of view of the side to move:
//         n < -100 : loss, but draw under 50-move rule
// -100 <= n < -1   : loss in n ply (assuming 50-move counter == 0)
//        -1        : loss, the side to move is mated
//         0        : draw
//     1 < n <= 100 : win in n ply (assuming 50-move counter == 0)
//   100 < n        : win, but draw under 50-move rule
//
// The return value n can be off by 1: a return value -n can mean a loss
// in n+1 ply and a return value +n can mean a win in n+1 ply. This
// cannot happen for tables with positions exactly on the "edge" of
// the 50-move rule.
//
// This implies that if dtz > 0 is returned, the position is certainly
// a win if dtz + 50-move-counter <= 99. Care must be taken that the engine
// picks moves that preserve dtz + 50-move-counter <= 99.
//
// If n = 100 immediately after a capture or pawn move, then the position
// is also certainly a win, and during the whole phase until the next
// capture or pawn move, the inequality to be preserved is
// dtz + 50-movecounter <= 100.
//
// In short, if a move is available resulting in dtz + 50-move-counter <= 99,
// then do not accept moves leading to dtz + 50-move-counter == 100.
int Tablebases::probe_dtz(Position& pos, ProbeState* result) {

    *result = OK;
    WDLScore wdl = search<true>(pos, result);

    if (*result == FAIL || wdl == WDLDraw) // DTZ tables don't store draws
        return 0;

    // DTZ stores a 'don't care' value in this case, or even a plain wrong
    // one as in case the best move is a losing ep, so it cannot be probed.
    if (*result == ZEROING_BEST_MOVE)
        return dtz_before_zeroing(wdl);

    int dtz = probe_table<DTZ>(pos, result, wdl);

    if (*result == FAIL)
        return 0;

    if (*result != CHANGE_STM)
        return (dtz + 100 * (wdl == WDLBlessedLoss || wdl == WDLCursedWin)) * sign_of(wdl);

    // DTZ stores results for the other side, so we need to do a 1-ply search and
    // find the winning move that minimizes DTZ.
    StateInfo st;
    int minDTZ = 0xFFFF;

    for (const Move& move : MoveList<LEGAL>(pos))
    {
        bool zeroing = pos.capture(move) || type_of(pos.moved_piece(move)) == PAWN;

        pos.do_move(move, st);

        // For zeroing moves we want the dtz of the move _before_ doing it,
        // otherwise we will get the dtz of the next move sequence. Search the
        // position after the move to get the score sign (because even in a
        // winning position we could make a losing capture or going for a draw).
        dtz = zeroing ? -dtz_before_zeroing(search(pos, result))
                      : -probe_dtz(pos, result);

        // If the move mates, force minDTZ to 1
        if (dtz == 1 && pos.checkers() && MoveList<LEGAL>(pos).size() == 0)
            minDTZ = 1;

        // Convert result from 1-ply search. Zeroing moves are already accounted
        // by dtz_before_zeroing() that returns the DTZ of the previous move.
        if (!zeroing)
            dtz += sign_of(dtz);

        // Skip the draws and if we are winning only pick positive dtz
        if (dtz < minDTZ && sign_of(dtz) == sign_of(wdl))
            minDTZ = dtz;

        pos.undo_move(move);

        if (*result == FAIL)
            return 0;
    }

    // When there are no legal moves, the position is mate: we return -1
    return minDTZ == 0xFFFF ? -1 : minDTZ;
}


// Use the DTZ tables to rank root moves.
//
// A return value false indicates that not all probes were successful.
bool Tablebases::root_probe(Position& pos, Search::RootMoves& rootMoves) {

    ProbeState result;
    StateInfo st;

    // Obtain 50-move counter for the root position
    int cnt50 = pos.rule50_count();

    // Check whether a position was repeated since the last zeroing move.
    bool rep = pos.has_repeated();

    int dtz, bound = Options["Syzygy50MoveRule"] ? 900 : 1;

    // Probe and rank each move
    for (auto& m : rootMoves)
    {
        pos.do_move(m.pv[0], st);

        // Calculate dtz for the current move counting from the root position
        if (pos.rule50_count() == 0)
        {
            // In case of a zeroing move, dtz is one of -101/-1/0/1/101
            WDLScore wdl = -probe_wdl(pos, &result);
            dtz = dtz_before_zeroing(wdl);
        }
        else
        {
            // Otherwise, take dtz for the new position and correct by 1 ply
            dtz = -probe_dtz(pos, &result);
            dtz =  dtz > 0 ? dtz + 1
                 : dtz < 0 ? dtz - 1 : dtz;
        }

        // Make sure that a mating move is assigned a dtz value of 1
        if (   pos.checkers()
            && dtz == 2
            && MoveList<LEGAL>(pos).size() == 0)
            dtz = 1;

        pos.undo_move(m.pv[0]);

        if (result == FAIL)
            return false;

        // Better moves are ranked higher. Certain wins are ranked equally.
        // Losing moves are ranked equally unless a 50-move draw is in sight.
        int r =  dtz > 0 ? (dtz + cnt50 <= 99 && !rep ? 1000 : 1000 - (dtz + cnt50))
               : dtz < 0 ? (-dtz * 2 + cnt50 < 100 ? -1000 : -1000 + (-dtz + cnt50))
               : 0;
        m.TBRank = r;

        // Determine the score to be displayed for this move. Assign at least
        // 1 cp to cursed wins and let it grow to 49 cp as the positions gets
        // closer to a real win.
        m.TBScore =  r >= bound ? VALUE_MATE - MAX_PLY - 1
                   : r >  0     ? Value((std::max( 3, r - 800) * int(PawnValueEg)) / 200)
                   : r == 0     ? VALUE_DRAW
                   : r > -bound ? Value((std::min(-3, r + 800) * int(PawnValueEg)) / 200)
                   :             -VALUE_MATE + MAX_PLY + 1;
    }

    return true;
}


// Use the WDL tables to rank root moves.
// This is a fallback for the case that some or all DTZ tables are missing.
//
// A return value false indicates that not all probes were successful.
bool Tablebases::root_probe_wdl(Position& pos, Search::RootMoves& rootMoves) {

    static const int WDL_to_rank[] = { -1000, -899, 0, 899, 1000 };

    ProbeState result;
    StateInfo st;

    bool rule50 = Options["Syzygy50MoveRule"];

    // Probe and rank each move
    for (auto& m : rootMoves)
    {
        pos.do_move(m.pv[0], st);

        WDLScore wdl = -probe_wdl(pos, &result);

        pos.undo_move(m.pv[0]);

        if (result == FAIL)
            return false;

        m.TBRank = WDL_to_rank[wdl + 2];

        if (!rule50)
            wdl =  wdl > WDLDraw ? WDLWin
                 : wdl < WDLDraw ? WDLLoss : WDLDraw;
        m.TBScore = WDL_to_value[wdl + 2];
    }

    return true;
}
