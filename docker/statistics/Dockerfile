FROM rocker/r-ver@sha256:c3f39b365d1077fe24f8e9ab2742e352b6d3950897f51af1624a5bb5550c21c0

ARG CRAN_SNAPSHOT=https://p3m.dev/cran/__linux__/noble/2026-04-23
ENV CRAN_SNAPSHOT=${CRAN_SNAPSHOT}

LABEL org.opencontainers.image.title="sslxray-statistics"
LABEL org.opencontainers.image.description="Frozen statistical runtime for SSOD chest X-ray S6+"
LABEL org.opencontainers.image.base.name="rocker/r-ver:4.5.3"
LABEL ssod.statistics.r_version="4.5.3"
LABEL ssod.statistics.cran_snapshot="2026-04-23"
LABEL ssod.statistics.base_digest="sha256:c3f39b365d1077fe24f8e9ab2742e352b6d3950897f51af1624a5bb5550c21c0"

RUN Rscript -e 'repo <- Sys.getenv("CRAN_SNAPSHOT"); pkgs <- c("lme4","lmerTest","pbkrtest","jsonlite","clubSandwich"); install.packages(pkgs, repos=repo, dependencies=NA); expected <- c(lme4="2.0.1",lmerTest="3.2.1",pbkrtest="0.5.5",jsonlite="2.0.0",clubSandwich="0.6.2"); for (p in names(expected)) { if (!requireNamespace(p, quietly=TRUE)) stop(paste("missing package",p)); observed <- as.character(utils::packageVersion(p)); cat(p,"=",observed,"\n",sep=""); if (observed != expected[[p]]) stop(paste("version mismatch",p,observed,expected[[p]])) }; cat("STATISTICAL_RUNTIME_BUILD_VERIFICATION=PASS\n")'
