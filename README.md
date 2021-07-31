# Music-Generation
An NLP project to generate novel music

### Requirements

- `tensorflow` `2.4.0`
- `selenium` to scrape?
- `scipy` to signal process if needed

### Major TODOs

- Implement a data input pipeline (dataset needs to be found for this - leaning towards dataset from [Mutopia](https://www.mutopiaproject.org/). Might have to scrape the site for the download links)
- Implement 1D causal convolutions (PixelCNNs) and hence the entire wavenet model
- Experiment with model sizes and hyperparams 


### Future

- Vary architecture, search if others work better
- Extend to speech generation?
