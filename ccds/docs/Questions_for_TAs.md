# Questions for TAs

### Q: What is the makefile?
A: 

---

### Q: Are we just downloading the data once, and consider it a static dataset, or is pulling data, part of our deployed pipeline?
A: we are expected to pull the data as part of the pipeline execution. The data is static, but were pretending this could be live data.

---

### Q: How do we get DVC working?
A: We had issues, getting `dvc pull` to work but this routine seems to do the trick.

 1) Download raw_data.csv manually from https://github.com/Jeppe-T-K/itu-sdse-project-data into data/raw

 2) Delete existing raw_data.csv.dvc 

 3) `dvc add raw_data.csv`

 4) `dvc pull`

---


