"""Allele-based orientation check for CopyKAT's tumour/normal calls (optional).

CopyKAT's final split is often right while its labels are swapped: when the normal reference
is wrong, the tumour cells are called diploid. Expression cannot settle which side is the tumour,
but alleles can: a copy-number change makes one parental haplotype over-represented across long
stretches of the genome, in tumour cells only, and no cell type can imitate that.

Pipeline (each step has a ``copykat-py-allele`` sub-command):

1. ``build-index`` (once): k-mers spanning known SNPs, both alleles, unique genome-wide.
2. ``count``: per-cell REF/ALT UMI counts from 10x reads on the GPU, without alignment
   (FASTQ.gz or SRA input). Cell Ranger BAMs can instead be piled up with cellsnp-lite; both
   write cellsnp-lite's output format.
3. ``panel`` (once) and ``phase``: phase the sample's heterozygous SNPs with Eagle2 against the
   1000 Genomes high-coverage panel.
4. ``orient``: a hidden Markov model over phased SNPs finds allelically imbalanced segments in
   each of CopyKAT's two groups; the calls are flipped if the "diploid" group carries more.

Requires reads (or a BAM), not just a count matrix. External tools: Eagle2, bcftools, and for
counting igzip (ISA-L) or fastq-dump, gcc, and a CUDA GPU with PyTorch.
Evidence: docs/research/allele_orientation.md.
"""

from copykat_py.allele.orient import orient_prediction  # noqa: F401
