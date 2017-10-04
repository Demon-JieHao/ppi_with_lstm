library(protr)
seqData <- readr::read_tsv("../data/sequences.fa.gz", col_names=FALSE)
names(seqData) <- c("uid", "sequence")
seqs <- seqData$sequence

### Leave only the canonical aminoacids
idx <- purrr::map_lgl(seqs, ~grepl('U', .x))
uid <- seqData$uid[!idx]
seqs <- seqs[!idx]

### Remove protein sequences that are either too short or too long
idx <- (nchar(seqs) >= 10) & (nchar(seqs) <= 500)
seqs <- seqs[idx]
uid <- uid[idx]

createFP <- function(seq) {
    c(
        extractAAC(seq),
        extractDC(seq),
        extractCTDC(seq),
        extractCTDT(seq),
        extractCTDD(seq),
        extractQSO(seq, nlag=25),
        extractAPAAC(seq, lambda = 25)
        ## extractMoreauBroto(seq),
        ## extractMoran(seq),
        ## qextractGeary(seq),
        ## extractCTriad(seq),
        ## extractSOCN(seq),
        ## extractPAAC(seq)
        ## extractPSSM(seq),
        ## extractPSSMAcc(seq),
        ## extractPSSMFeature(seq),
        ## extractScales(seq),
        ## extractProtFP(seq),
        ## extractProtFPGap(seq),
        ## extractDescScales(seq),
        ## extractFAScales(seq),
        ## extractMDSScales(seq),
        # extractBLOSUM(seq)
    )
}

proteinFPs <- lapply(seqs, createFP)
names(proteinFPs) <- uid
save(proteinFPs, file = "../output/proteinFP.RData")
