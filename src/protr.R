library(protr)
seqData <- readr::read_tsv("../data/sequences.fa.gz", col_names=FALSE)
names(seqData) <- c("uid", "sequence")

### Leave only the canonical aminoacids
idx <- sapply(seqData$sequence, protcheck)
seqData <- seqData[idx, ]

### Remove protein sequences that are either too short or too long
## seqLen <- sapply(seqData$sequence, nchar)
## idx <- (seqLen >= 50) & (seqLen <= 1000)
## seqData <- seqData[idx, ]

createFP <- function(seq) {
    c(
        extractAAC(seq),
        extractDC(seq),
        extractCTDC(seq),
        extractCTDT(seq),
        extractCTDD(seq),
        extractQSO(seq, nlag = 20),
        extractAPAAC(seq, lambda = 20),
        extractMoreauBroto(seq, nlag = 20),
        extractMoran(seq, nlag = 20),
        extractGeary(seq, nlag = 20),
        extractCTriad(seq),
        extractSOCN(seq, nlag = 20),
        extractPAAC(seq, lambda = 20),
        extractScales(seq,
                      propmat=t(na.omit(as.matrix(AAindex[, 7:26]))),
                      pc = 5, lag = 7),
        extractProtFP(seq, index = c(160:165, 258:296),
                      pc = 5, lag = 7),
        extractProtFPGap(seq, index = c(160:165, 258:296),
                         pc = 5, lag = 7),
        extractDescScales(seq, propmat = "AATopo",
                          index = c(37:41, 43:47),
                          pc = 5, lag = 7),
        extractFAScales(seq, propmat = AATopo[, c(37:41, 43:47)],
                        factors = 5, lag = 7),
        extractMDSScales(seq, propmat = AATopo[, c(37:41, 43:47)],
                         k = 5, lag = 7),
        extractBLOSUM(seq, submat = "AABLOSUM62",
                      k = 5, lag = 7, scale = TRUE)
    )
}

proteinFPs <- lapply(seqData$sequence, createFP)
names(proteinFPs) <- seqData$uid
proteinFPs <- t(as.data.frame(proteinFPs))
### Note that now proteinFPs is a matrix
save(proteinFPs, file = "../output/proteinFP.RData")
### You need a data frame for write_tsv
proteinFPs <- data.frame(uid = rownames(proteinFPs), proteinFPs,
                         stringsAsFactors = FALSE)
readr::write_tsv(proteinFPs, path = "../output/proteinFPs.tsv")
