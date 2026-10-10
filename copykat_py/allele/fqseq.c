/* fqseq: read FASTQ on stdin, write only the sequence lines (line 2 of every 4-line record), each
   truncated or N-padded to exactly L bytes with no newline, so the output is a dense (n, L) array.
   Usage: fqseq L < reads.fastq > seqs.bin */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define IN (1 << 24)

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: fqseq L\n"); return 2; }
    long L = atol(argv[1]);
    char *in = malloc(IN), *out = malloc(IN + L + 1);
    size_t have = 0, o = 0;
    long line = 0;            /* line number modulo 4 of the line starting at the cursor */
    for (;;) {
        ssize_t r = read(0, in + have, IN - have);
        if (r < 0) return 1;
        size_t n = have + (size_t)r;
        size_t pos = 0;
        for (;;) {
            char *nl = memchr(in + pos, '\n', n - pos);
            if (!nl) break;
            size_t len = (size_t)(nl - (in + pos));
            if (line == 1) {
                size_t c = len < (size_t)L ? len : (size_t)L;
                memcpy(out + o, in + pos, c);
                if (c < (size_t)L) memset(out + o + c, 'N', (size_t)L - c);
                o += (size_t)L;
                if (o > IN) { fwrite(out, 1, o, stdout); o = 0; }
            }
            line = (line + 1) & 3;
            pos += len + 1;
        }
        have = n - pos;
        memmove(in, in + pos, have);
        if (r == 0) break;
    }
    fwrite(out, 1, o, stdout);
    return 0;
}
