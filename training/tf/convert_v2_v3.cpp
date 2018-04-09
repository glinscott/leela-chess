// g++ convert_v2_v3.cpp -std=c++14 -O3 -o convert -lz

#include <iostream>
#include <fstream>
#include <vector>

#include <zlib.h>


static constexpr int NUM_HIST = 8;
static constexpr int NUM_PIECE_TYPES = 6;

struct v2_t {
  int32_t version;
  float probs[1924];
  uint64_t planes[112];
  uint8_t us_ooo;
  uint8_t us_oo;
  uint8_t them_ooo;
  uint8_t them_oo;
  uint8_t stm;
  uint8_t r50;
  uint8_t cnt;
  int8_t result;
} __attribute__((packed));

struct v3_t {
  int32_t version;
  float probs[1858];
  uint64_t planes[104];
  uint8_t us_ooo;
  uint8_t us_oo;
  uint8_t them_ooo;
  uint8_t them_oo;
  uint8_t stm;
  uint8_t r50;
  uint8_t cnt;
  int8_t result;
} __attribute__((packed));


uint64_t reverse(uint64_t x) {
  uint64_t r;
  auto a = reinterpret_cast<char*>(&x);
  auto b = reinterpret_cast<char*>(&r);

  for (int i = 0; i < 4; i++) {
    b[i] = a[7-i];
    b[7-i] = a[i];
  }

  return r;
}


int main(int argc, char **argv) {
  static_assert(sizeof(v2_t) == 8604, "invalid v2");
  static_assert(sizeof(v3_t) == 8276, "invalid v3");

  if (argc != 3) {
    std::cout << "usage: " << argv[0] << " in.gz out.gz" << std::endl;
    return 1;
  }

  gzFile infile = gzopen(argv[1], "rb");
  gzFile outfile = gzopen(argv[2], "wb");

  v2_t v2;
  v3_t v3;
  v3.version = 3;
  v3.cnt = 0;
  int count = 0;

  while (gzread(infile, reinterpret_cast<char*>(&v2), sizeof(v2)) > 0) {
    if (v2.version != 2) {
      throw std::runtime_error("Invalid version");
    }

    // probabilities

    // planes
    for (int i = 0; i < NUM_HIST; i++) {
      for (int j = 0; j < NUM_PIECE_TYPES; j++) {
        if (i&1) {
          v3.planes[i*13+j] = reverse(v2.planes[i*14+j]);
        }
        else {
          v3.planes[i*13+j] = v2.planes[i*14+j];
        }
      }
    }

    v3.us_ooo = v2.us_ooo;
    v3.us_oo = v2.us_oo;
    v3.them_ooo = v2.them_ooo;
    v3.them_oo = v2.them_oo;
    v3.stm = v2.stm;
    v3.r50 = v2.r50;
    v3.result = v2.result;
    gzwrite(outfile, reinterpret_cast<char*>(&v3), sizeof(v3));
    count++;
  };

  gzclose(infile);
  gzclose(outfile);

  return 0;
}
