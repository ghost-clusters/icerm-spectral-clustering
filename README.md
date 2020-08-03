# Summer@ICERM 2020 - Spectral Clustering

Team name: Ghost Clusters

Non-Euclidean metrics? Who ya gonna call?


### Organization of the Repository
Some style guidelines:
- All data matrices should use row vectors to represent data, so that rows index datapoints and columns index features
- All lib scripts use lowercase titles
```
environment.yml
*.ipynb [For NBs under active development, they should live in the TLD]
lib/
    - Only code having tests and comments
    data/
        - For code interacting with data, this directory holds data flatfiles
scripts/
    - Testing ground, poor code is ok
artifacts/
    - archive for experimental results or other hard copies of things
    - ideally, only .png plots, .pdf notes, anything which no longer needs access to lib/
    - for expts that do need lib/ (such as notebooks), make a copy of lib/ and store with the experiment
    exp_1/
    exp_2/ 
    ...
```
