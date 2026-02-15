# Legal Paraphrase Mining

Motivation: mining paraphrase pairs from legal text for semantic search and document comparison.

By Thomas Decloedt

# Intro

Goal: obtain pairs of sentences which differ in terms of the words they use but have near identical or very similar meaning.

Input: a corpus of legal documents.

# TL;DR

1. Installation: `./install.sh`
2. Mining: `uv run python -m src.tiny_test`
3. Translation: `uv run python -m src.synthetic_pair_creation`

# System

## Design choices

The basic assumption behind my design choices, which I found to be reasonable, is the following:
truly interesting paraphrases are to be found in content written by humans, especially in a specialized domain,
and above all if it was written by experts.

The dataset fits this.
Hence, an approach that is mainly based around mining rather than generation was chosen.
To this end, I mainly optimized the mining pipeline for speed, wherever possible.
Optimizing for precision above recall, simply because:
finding 95% of available paraphrases, say 10k
is still less than finding 50% of 100k available sentences.

The computationally most expensive operation is densely embedding text, hence this was only kept as final quality check.
Everything before was done using keywords and sparse embeddings.
The philosophy was to pre-filter quite aggressively beforehand and only then do semantic similarity thresholding.

I ranked the found paraphrases based on the following quality heuristic function $q$:

$$
q(s_1, s_2) = f_{\text{cos}}(s_1, s_2) \times [ 1 - f_{\text{jac}}(s_1, s_2) ]
$$

where $f_{\text{cos}}$ and $f_{\text{jac}}$ are cosine and Jaccard similarity, respectively.
The former indicates how similar the two sentences are in meaning, the latter how lexically similar they are.
This reflects one of the core quality aspects: that the variation in words is large while variation in meaning is small.

## Frameworks

**Qdrant**: Vector database used for efficient sparse embedding storage and approximate nearest neighbor (ANN) search. Chosen for its native support of sparse vectors and high performance at scale.

**FastEmbed (BM25)**: Provides BM25 sparse text embeddings for keyword-based initial retrieval. Much faster than dense embeddings and provides good precision for finding paraphrase candidates.

**Jina Embeddings v3**: Dense embedding model (via sentence-transformers) for final verification of candidates after aggressive pre-filtering. Only applied to a small subset to keep computational costs manageable.

**NLTK**: WordNet for synonym expansion and SnowballStemmer for morphological normalization (English and German).

**spaCy**: Initially tried for linguistic analysis but found it too slow for processing large corpora. Fell back to regex-based approaches for entity extraction.

**syntok**: Sentence splitter chosen over simple regex for robustness (handles edge cases like "Prof.", "Art.", etc.).

## Details

### Keyword extraction and stemming

Keywords are extracted using a fast regex-based approach (much faster than spaCy) with stopword filtering.
To improve recall across morphological variations, both the original word and its stemmed form are included.

**Stopwords**: Combines spaCy's default stopwords with domain-specific legal stopwords:
- **English**: Custom EU legal stopwords ([src/stopwords/EU_legal_EN.csv](src/stopwords/EU_legal_EN.csv)) covering procedural terms (acting, orders, dismissed), boilerplate (pursuant, whereas, hereby), and legal discourse markers
- **German**: SW-DE-RS v1.0.0 from [Zenodo](https://zenodo.org/records/3995593) (CC0 license), based on high-frequency words from German Federal Courts (1998-2020)

**Stemming**: Uses NLTK's SnowballStemmer for both English and German. For example, "diagnosed", "diagnosing", "diagnosis" all stem to "diagnos", allowing matches across different word forms.

**Bigrams**: Also extracts consecutive word pairs (bigrams) from both original and stemmed tokens to capture multi-word expressions like "mental illness".

### Synonym expansion

To improve recall without sacrificing the speed advantages of keyword-based retrieval, extracted keywords are expanded with synonyms before querying the vector database. This allows finding paraphrases that use different but semantically related terms.

**English**: Uses NLTK's WordNet to generate up to 5 synonyms per keyword. For example, "mental" might expand to include "psychological", "psychiatric", etc.

**German**: Uses OdeNet (Open German WordNet) for synonym generation.

The expansion is limited (max 5 synonyms per keyword) to keep the search space manageable and avoid noise from overly broad synonyms. Synonyms are also stemmed to maximize matching potential.

### Entity filter

Extracted using regex some recurring entities and removed candidates early before embedding when not matching, for example:
* Under Article 87 of the Rules of Procedure of the Court of First Instance...
* Under Article 78 of the Rules of Procedure of the Court of First Instance...
Reasoning: in the legal domain a different article (or similar entity) has a huge implication for the meaning of the sentence,
I assumed no expert would thus see them as paraphrases.

## Approaches and Trade-offs

### Mining vs Construction

**Mining approach (chosen)**: Retrieve paraphrase candidates directly from the corpus using keyword-based similarity.
- Higher lexical diversity from real-world expert writing
- Quality: sentences written by legal experts

**Construction approach (rejected)**: Generate paraphrases via keyword extraction, MLM-based lexical substitution, and syntactic transformations.
- Slow speed: keyword extraction + masking + parsing for each sentence
- Low lexical diversity despite good meaning preservation

Example.
* Input: The applicant’s mental illness had been diagnosed on many occasions.
* Keyword: mental
* Substitution using LEGAL-BERT, incidentally trained on the same corpus:
    * Input: The applicant’s mental [MASK] had been diagnosed on many occasions.
    * Output: { personality, psychiatric, spinal, mental, anxiety, etc. }
* Parsing and transformation:
    1. The applicant’s mental illness had been diagnosed on many occasions.
    2. `{The, DET, definite_article} {applicant, N, common, singular} {'s, POSS, clitic} {mental, ADJ} {illness, N, common, singular} {had, AUX, past} {been, AUX, past_participle} {diagnosed, V, past_participle, passive} {on, PREP} {many, DET, quantifier} {occasions, N, common, plural} {., PUNCT, sentence_final}`
    3. `{On, PREP} {many, DET} {occasions, N} {,, PUNCT} {the, DET} {applicant, N} {'s, POSS} {mental, ADJ} {illness, N} {had, AUX} {been, AUX} {diagnosed, V} {., PUNCT}`

We might get a sentence like this:
On many instances, the person's psychological condition had been confirmed.

Same meaning but not all that different in wording.

### Retrieval Strategy: Sparse vs Dense Embeddings

**BM25 (sparse) for initial retrieval**: fast, scalable, good precision for finding paraphrase candidates

**Dense embeddings only for final verification**: too expensive to embed all sentences upfront, used only after aggressive pre-filtering

**Why not full dense embedding approach**:
- Doesn't scale to 100M+ sentences (too slow even with faster embedding models)
- Even fast static embedders (Model2Vec) are slower than keyword-based approaches for mono-lingual retrieval
- Fast static embedders lack cross-lingual alignment: given sentence $S$ and translation $S'$, computing $f_{cos}(S, S')$ leads to very low similarity, making them unusable for cross-lingual mining

### Cross-lingual Strategy

**Some multilingual paraphrases found naturally**: The default BM25 sparse embedding pipeline finds some cross-lingual pairs without special handling.

**Translation-based keyword expansion (tried and rejected)**:
- Uses word2word bilingual dictionaries to translate keywords before sparse embedding retrieval
- Problem: generates too many low-quality candidates, slowing down the pipeline due to more candidates requiring expensive dense embedding verification
- Many candidates get filtered out, making the approach inefficient
- Would need more experimentation with alternative approaches that are more precise and non-dictionary based

**Synthetic translation generation (final approach)**:
- Separate translation step to create cross-lingual pairs after mining
- Dense embedding-based filtering to remove poor quality synthetic paraphrases
- **Known limitation**: lower quality due to translation model accuracy, syntactically incorrect sentences, requires better cleaning for production use

## Results

**Corpus**: sample from Multi_Legal_Pile of 20M sentences (10M English, 10M German)

**Mined paraphrases**: ~99K pairs
- ~39K pairs with semantic similarity > 0.9
- Cross-lingual (EN-DE) with high lexical diversity
- Runtime: ~4 hours (most time spent on dense embedding verification; too many candidates, would need parameter tuning to reduce candidates further)

**Synthetic paraphrases**: ~362K pairs
- Lower quality, requires production cleaning
- Runtime: ~3 hours (most time spent on dense embedding verification; a better translation model and no verification could be better)

**Known limitations**: synthetic pairs contain syntactic errors and translation artifacts

**Output files**: Both result sets are available in [results/paraphrase_pairs.zip](results/paraphrase_pairs.zip):
- `mined_paraphrases.csv` - ~99K mined paraphrase pairs with quality scores
- `synthetic_cross_lingual_paraphrases.csv` - ~362K synthetic cross-lingual pairs
