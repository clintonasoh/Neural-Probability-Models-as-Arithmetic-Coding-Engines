"""
arithmetic_coder.py
-------------------
A precision-safe, integer-based arithmetic encoder and decoder.

Uses 32-bit scaled integer arithmetic with E1/E2/E3 rescaling (the standard
Witten-Neal-Cleary approach) to avoid underflow without floating-point drift.

The encoder and decoder both accept symbol probabilities as integer counts
(numerators summing to a given total), so they can work with any probability
model — static frequency tables, adaptive models, or LLM softmax outputs.
"""

PRECISION     = 32
FULL          = 1 << PRECISION
HALF          = 1 << (PRECISION - 1)
QUARTER       = 1 << (PRECISION - 2)
THREE_QUARTER = 3 * QUARTER


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ArithmeticEncoder:
    """
    Encodes a sequence of symbols into a compact bitstream.

    Usage:
        enc = ArithmeticEncoder()
        for symbol_index in sequence:
            enc.encode(cdf[symbol_index], cdf[symbol_index + 1], total)
        bits = enc.finish()
    """

    def __init__(self):
        self.low     = 0
        self.high    = FULL
        self.bits    = []
        self.pending = 0  # pending bits from E3 rescaling

    def encode(self, cum_low: int, cum_high: int, total: int):
        """Encode one symbol given its CDF range [cum_low, cum_high) / total."""
        assert 0 <= cum_low < cum_high <= total, \
            f"Invalid CDF: [{cum_low}, {cum_high}) / {total}"

        rng       = self.high - self.low
        self.high = self.low + (rng * cum_high) // total
        self.low  = self.low + (rng * cum_low)  // total

        if self.low >= self.high:
            self.high = self.low + 1

        self._rescale()

    def _rescale(self):
        while True:
            if self.high <= HALF:
                self._emit(0)
                self.low  *= 2
                self.high *= 2
            elif self.low >= HALF:
                self._emit(1)
                self.low  = (self.low  - HALF) * 2
                self.high = (self.high - HALF) * 2
            elif self.low >= QUARTER and self.high <= THREE_QUARTER:
                self.pending += 1
                self.low  = (self.low  - QUARTER) * 2
                self.high = (self.high - QUARTER) * 2
            else:
                break

    def _emit(self, bit: int):
        self.bits.append(bit)
        for _ in range(self.pending):
            self.bits.append(1 - bit)
        self.pending = 0

    def finish(self) -> list:
        """Flush remaining state; return the complete bit list."""
        self.pending += 1
        if self.low < QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        return self.bits


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class ArithmeticDecoder:
    """
    Decodes a bitstream produced by ArithmeticEncoder.

    Usage:
        dec = ArithmeticDecoder(bits)
        for each position:
            symbol = dec.decode(cdf, total)
    """

    def __init__(self, bits: list):
        self.bits  = bits
        self.pos   = 0
        self.low   = 0
        self.high  = FULL
        self.value = 0
        for _ in range(PRECISION):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self.pos < len(self.bits):
            b = self.bits[self.pos]
            self.pos += 1
            return b
        return 0

    def decode(self, cdf: list, total: int) -> int:
        """
        Decode one symbol. `cdf` is a list of len(vocab+1) cumulative counts.
        Returns the decoded symbol index.
        """
        rng    = self.high - self.low
        scaled = ((self.value - self.low + 1) * total - 1) // rng

        # Binary search for the symbol
        lo, hi = 0, len(cdf) - 2
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid + 1] <= scaled:
                lo = mid + 1
            else:
                hi = mid
        symbol = lo

        self.high = self.low + (rng * cdf[symbol + 1]) // total
        self.low  = self.low + (rng * cdf[symbol])     // total

        if self.low >= self.high:
            self.high = self.low + 1

        self._rescale()
        return symbol

    def _rescale(self):
        while True:
            if self.high <= HALF:
                self.low   = self.low  * 2
                self.high  = self.high * 2
                self.value = self.value * 2 + self._read_bit()
            elif self.low >= HALF:
                self.low   = (self.low  - HALF) * 2
                self.high  = (self.high - HALF) * 2
                self.value = (self.value - HALF) * 2 + self._read_bit()
            elif self.low >= QUARTER and self.high <= THREE_QUARTER:
                self.low   = (self.low  - QUARTER) * 2
                self.high  = (self.high - QUARTER) * 2
                self.value = (self.value - QUARTER) * 2 + self._read_bit()
            else:
                break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bits_to_bytes(bits: list) -> bytes:
    """Pack a bit list into bytes (zero-padded to next multiple of 8)."""
    padded = bits + [0] * (-len(bits) % 8)
    out = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for b in padded[i:i+8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


def bytes_to_bits(data: bytes) -> list:
    """Unpack bytes back into a bit list."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def build_cdf(counts: list) -> tuple:
    """
    Build an integer CDF from raw counts (with +1 Laplace smoothing).
    Returns (cdf_list, total).
    """
    smoothed = [c + 1 for c in counts]
    total    = sum(smoothed)
    cdf      = [0] * (len(smoothed) + 1)
    for i, c in enumerate(smoothed):
        cdf[i + 1] = cdf[i] + c
    return cdf, total
