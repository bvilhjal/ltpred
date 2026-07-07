# Real-LD GWAS benchmark with HAPNEST genotypes (opt-in)

`bench_gwas_power.py` simulates independent SNPs by default — enough for the
power comparison, which turns on each phenotype's correlation to the true genetic
value, not on LD. To score the methods on **realistic LD** (so the multiple-
testing structure and λ_GC calibration reflect a real genome), feed it genotypes
from [**HAPNEST**](https://github.com/intervene-EU-H2020/synthetic_data) (Wharrie
et al., *Bioinformatics* 2023): synthetic genotypes **resampled from a real
1000G + HGDP reference**, carrying real LD, MAF spectra and population structure.

HAPNEST produces genotypes for **unrelated** individuals, which is exactly what
this benchmark needs: the LT-FH++/PA-FGRS family history is *simulated on top* of
each proband's real-LD genotype (relatives' liabilities are drawn conditional on
the genotype-derived genetic liability). So HAPNEST supplies the probands' real
genomes; ltpred supplies the pedigree and the liability-threshold phenotype.

## Step 1 — generate genotypes with HAPNEST

HAPNEST is a container-distributed tool; the reference download is multi-GB, so
this is an opt-in local step (not CI). See the config here as a starting point.

```bash
singularity pull docker://sophiewharrie/intervene-synthetic-data
mkdir -p data/output
cp benchmarks/hapnest/config_ltfh.yaml data/config.yaml     # then edit paths

SIF=intervene-synthetic-data_latest.sif
singularity exec --bind data/:/data/ $SIF init      # writes an example config to diff against
singularity exec --bind data/:/data/ $SIF fetch     # downloads the 1000G+HGDP reference (large)
singularity exec --bind data/:/data/ $SIF generate_geno 4 data/config.yaml
```

This writes a PLINK fileset (`.bed/.bim/.fam`) under `data/output/`. Only the
**genotypes** are used — the phenotype is simulated by the benchmark under the
liability-threshold model, so `generate_pheno` is not needed. **Reconcile
`config_ltfh.yaml` against the `init`-generated example first**; HAPNEST's field
names are stable but the YAML nesting can shift between versions.

## Step 2 — run the benchmark on the real-LD genotypes

```bash
OMP_NUM_THREADS=10 python benchmarks/bench_gwas_power.py \
    --plink data/output/synthetic \
    --n-fam 10000 --n-causal 30 --h2 0.5 --prev 0.05
```

`--plink PREFIX` loads the fileset via the minimal reader in `_common.py`
(SNP-major `.bed`, missing calls mean-imputed) and uses the first `--n-fam`
individuals as probands. Everything else is identical to the default run:
causal effects are drawn on the real genotypes, family history is simulated, and
the four phenotypes (case/control, LT-FH++, PA-FGRS, oracle) are scored on mean
χ² at causal SNPs, detection power, and λ_GC at nulls — now with real LD under
the nulls.

## Caveats

- **Not CI-runnable**: the reference download and generation are large and slow.
- The benchmark reads only `.bed/.bim/.fam`; it never parses a HAPNEST phenotype
  file (the phenotype is the simulated liability-threshold family history).
- With real LD the causal SNPs tag neighbours, so `power` counts tagged signals
  too; the mean-χ²-at-causal and eff-N ratios remain the cleanest method contrast.
