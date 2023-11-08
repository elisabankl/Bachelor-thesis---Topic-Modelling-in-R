20Newsgroups
================
Elisa Bankl
2023-10-01

Load all necessary packages.

``` r
library(tm)
```

    ## Lade nötiges Paket: NLP

``` r
library(topicmodels)
library(quanteda)
```

    ## Warning in .recacheSubclasses(def@className, def, env): Nicht definierte
    ## Subklasse "pcorMatrix" von Klasse "xMatrix"; Definition nicht aktualisiert

    ## Warning in .recacheSubclasses(def@className, def, env): Nicht definierte
    ## Subklasse "pcorMatrix" von Klasse "mMatrix"; Definition nicht aktualisiert

    ## Warning in .recacheSubclasses(def@className, def, env): Nicht definierte
    ## Subklasse "pcorMatrix" von Klasse "replValueSp"; Definition nicht aktualisiert

    ## Package version: 3.3.1
    ## Unicode version: 13.0
    ## ICU version: 69.1

    ## Parallel computing: 4 of 4 threads used.

    ## See https://quanteda.io for tutorials and examples.

    ## 
    ## Attache Paket: 'quanteda'

    ## Das folgende Objekt ist maskiert 'package:tm':
    ## 
    ##     stopwords

    ## Die folgenden Objekte sind maskiert von 'package:NLP':
    ## 
    ##     meta, meta<-

``` r
library(tidyr)
library(dplyr)
```

    ## 
    ## Attache Paket: 'dplyr'

    ## Die folgenden Objekte sind maskiert von 'package:stats':
    ## 
    ##     filter, lag

    ## Die folgenden Objekte sind maskiert von 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(pheatmap)
library(RColorBrewer)
library(LDAvis)
library(LSAfun)
```

    ## Lade nötiges Paket: lsa

    ## Lade nötiges Paket: SnowballC

    ## Lade nötiges Paket: rgl

Specify the location of the folder with the documents in the 20
Newsgroups dataset.

## Load data, create corpus

``` r
directory_location = "C:/Users/elisa/Documents/VWA-2/20news-18828.tar/20news-18828/20news-18828"
```

``` r
corpus_source = DirSource(directory = directory_location,recursive = TRUE)
```

The newsgroup categories are in the names of the subfolders create a
vector with the newsgroups the same length as the number of documents.

``` r
subfolder_names <- sapply(corpus_source$filelist, function(path) {
  basename(dirname(file.path(directory_location, path)))
})
subfolder_names = unname(subfolder_names)
```

Create a volatile corpus.

``` r
corpus = VCorpus(corpus_source)
```

Remove the ‘’From:’ tags from the documents, to remove information that
might the newsgroups.

``` r
remove_from_tag <- content_transformer(function(x) {
  gsub("^From:.*", "", x)
})

corpus <- tm_map(corpus, remove_from_tag)
```

Shuffle corpus and metadata because the documents are sorted by
newsgroup.

``` r
set.seed(123)  # Set a random seed for reproducibility
shuffle_indices <- sample(length(corpus))
corpus <- corpus[shuffle_indices]
subfolder_names = subfolder_names[shuffle_indices]
```

## Preprocessing

Many of the steps could be also done while creating the document term
matrix

Parts of the code were taken from
<https://ladal.edu.au/topicmodels.html> on \[12-10-2023\].

``` r
#set to lower case
processedCorpus <- tm_map(corpus, content_transformer(tolower))

#remove email adresses
remove_words_with_at <- function(x) {
  words <- unlist(strsplit(x, " "))  # Split the text into words
  words <- words[!grepl("@", words)]    # Remove words containing '@'
  cleaned_text <- paste(words, collapse = " ")  # Recombine the words into a single string
  return(cleaned_text)
}

processedCorpus = tm_map(processedCorpus, content_transformer(remove_words_with_at))
#remove stopwords and the names of the newsgroups
processedCorpus <- tm_map(processedCorpus, removeWords, c(stopwords(),unique(subfolder_names)))
#remove numbers
processedCorpus <- tm_map(processedCorpus, removeNumbers)
#remove punctuation
processedCorpus <- tm_map(processedCorpus, removePunctuation, preserve_intra_word_dashes = TRUE)

#stem
processedCorpus <- tm_map(processedCorpus, stemDocument, language = "en")
#strip white spaces
processedCorpus <- tm_map(processedCorpus, stripWhitespace)
```

Create a Document Term Matrix

By default DTM uses the ‘words’ tokenizer. Take only words that appear
in least 15 document and at most 80% of documents in the corpus.

``` r
DTM <- tm::DocumentTermMatrix(processedCorpus, control = list(bounds = list(global = c(15, length(processedCorpus)*0.8))))
```

Due to vocabulary pruning, there are empty rows in the DTM. We remove
the empty documents from the DTM and also delete the corresponding
metadata.

``` r
sel_indices <- slam::row_sums(DTM) > 0
DTM <- DTM[sel_indices, ]
subfolder_names <- subfolder_names[sel_indices]
corpus <- corpus[sel_indices] #needed for visualizing the documents
processedCorpus = processedCorpus[sel_indices] #needed for visualizing the stemmed doucments
```

## LDA

### Fit LDA model

fit LDA using Gibbs Sampling

``` r
set.seed(523)
topicModel_lda <- topicmodels::LDA(DTM, 20, method="Gibbs", control=list(iter=2000,thin=2000,initialize = "random",best=TRUE))
```

### Create heatmap

convert the matrix theta into a dataframe.

``` r
theta_frame= data.frame(topicModel_lda@gamma)
```

Try to imitate the way LDAvis determines the topic order. To do this, we
multiply the matrix $\theta$ of document-topic-distribution with a
vector of the document lengths.

``` r
topic_prevalence <- slam::row_sums(DTM, na.rm = TRUE)%*%(topicModel_lda@gamma) 
order = order(order(-topic_prevalence)) #order the topics by prevelance in the corpus
topic_rank = colnames(theta_frame)
names(topic_rank) <- order #map the document names to the rank of the document
```

Get a dataframe that contains the average $\theta$ per topic and
newsgroup in one column.

``` r
theta_wide <- tibble(theta_frame,'newsgroup'=unlist(subfolder_names))

theta_wide %>% 
  group_by(newsgroup) %>%
  summarise_if(is.numeric, mean, na.rm = TRUE)%>%
  tibble::remove_rownames() %>% 
  tibble::column_to_rownames(var="newsgroup")->theta_average
theta_average <- rename(theta_average, all_of(topic_rank)) #change the column names to the rank of the topic
```

Create heatmap of the average $\theta$ for every topic and newsgroup and
save it to png file.

``` r
#png("../heatmap.png", width = 700*300/100,height = 500*300/100,res=300)
pheatmap(
  as.matrix(
      theta_average
    ), 
  color = colorRampPalette(brewer.pal(n = 7, name ="YlGnBu"))(100),
  border_color = NA,scale ='none',angle_col=0)
```

![](20Newsgroups-from-directory_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
#dev.off()
```

### Print out documents with words colored according to their topic assignment in z.

Create a DTM with the unpreprocessed tokens, to get the unique tokens.
This could be done more elegantly.

``` r
dtm_unprocessed <- DocumentTermMatrix(corpus,control = list(removePunctuation=TRUE,tolower=TRUE,bounds =list(global=c(15,Inf))))
```

Map the unprocessed tokens to processed tokens.

``` r
# Get the column names (words) from the unprocessed DTM
words <- colnames(dtm_unprocessed)

# Define preprocessing steps
preprocess_word <- function(word) {
  # Apply preprocessing steps to each word
  word <- tolower(word)  
  if ("@" %in% word){
    return("")
  }
  if (word %in%stopwords()){
    return("")
  }
  word = removeNumbers(word)# Remove numbers
  word = removePunctuation(word, preserve_intra_word_dashes = TRUE)# Convert to lowercase
  word = stripWhitespace(word)
  
  # Stem the word
  word <- stemDocument(word)
  return(word)
}
# Apply preprocessing steps to the words and create a named list
word_mapping <- setNames(lapply(words, preprocess_word), words)
```

Create 2 lists of text colors in latex. The first list is used for to
color text, the second list is used to color the background of the

``` r
#list of colors that can be used to color text in latex
list_of_latex_colors1 = c("red","green","blue","cyan","magenta","gray","teal","violet",'brown','purple','orange','pink','olive')
list_of_latex_colors2 = c("red","green","cyan","magenta","gray","teal",'lime','brown','purple','orange','pink','olive')
```

Map the words to the indices of the processed word in the term list
returned by LDA.

``` r
words_to_indices <- match(unlist(word_mapping), topicModel_lda@terms, nomatch = NULL)
words_to_indices <- as.list(words_to_indices)
names(words_to_indices) <- names(word_mapping)
```

Create a dataframe with z and the corresponding document and word index.
This assumes that z is filled going through the document term matrix row
by row (ie document by document) and adding for every column (ie for
every term) as many entries as it’s value. This means that if a word
appears three times in a document, for this word and document, three
consecutive entries are added to z.

``` r
row_indices <- rep(DTM$i, DTM$v)
col_indices <- rep(DTM$j, DTM$v)
z_frame = data.frame(rows = row_indices,cols = col_indices,topic = topicModel_lda@z,accessed=0)
```

``` r
doc_number_list <- c(6234,17121,1729)
subfolder_names[doc_number_list]
```

    ## [1] "alt.atheism"     "sci.space"       "rec.motorcycles"

``` r
for(doc_number in doc_number_list){

#find the unprocessed words in the selected document by splitting the document at white spaces
list_of_words_in_doc <- NLP::words(stripWhitespace(as.character(corpus[[doc_number]])), " ")
#map the words to their most probable topic in the specific document (using @wordassignments)


#get the right row of the @wordassignments matrix
row_values <- topicModel_lda@wordassignments[doc_number, ]
#put the values (=most probable topic assignments for the words) in a list
topic_word_map <- unlist(row_values$v)
#the names of the list are the indices of the words
names(topic_word_map) <- unlist(row_values$j)
#create a list with the most probable topic assignments in the corpus

#first get a list with the word indices of the words in the document
topic_list = words_to_indices[removePunctuation(tolower(list_of_words_in_doc))]
#replace NULLs by NAs
topic_list <- lapply(topic_list, function(x) if (is.null(x)) NA else x)
#now get the topics for the words, using the word indices and the topic_word_map 
#obtained from @wordassignments
topic_list = topic_word_map[as.character(unlist(topic_list))]

#print out the latex code for the words in the document colored according to their most probable topic assignment

#sink(file = '../colortext/'+String(doc_number)+'.tex')
vect=c()
for(i in 1:length(list_of_words_in_doc)){
if(is.na(topic_list[[i]])){
append(vect,cat("\\textcolor{",'black','}{',list_of_words_in_doc[[i]],'} ',sep=""))}
else if(topic_list[[i]]<=length(list_of_latex_colors1)){
  append(vect,cat("\\textcolor{",list_of_latex_colors1[topic_list[[i]]],'}{',list_of_words_in_doc[[i]],'} ',sep=""))
}
else {
append(vect,cat("\\colorbox{",list_of_latex_colors2[topic_list[[i]]-length(list_of_latex_colors1)],'}{',list_of_words_in_doc[[i]],'} ',sep=""))}
}
#sink(file=NULL)

}
```

    ## \textcolor{black}{Subject:} \textcolor{black}{Re:} \textcolor{black}{?} \textcolor{black}{(was} \textcolor{black}{Re:} \colorbox{cyan}{"Cruel"} \textcolor{black}{(was} \textcolor{black}{Re:} \colorbox{cyan}{<Political} \textcolor{brown}{Atheists?))} \textcolor{black}{sdoe@nmsu.edu} \colorbox{magenta}{(Stephen} \colorbox{lime}{Doe)} \textcolor{pink}{writes:} \textcolor{black}{>>Of} \textcolor{brown}{course,} \textcolor{black}{if} \textcolor{black}{at} \textcolor{black}{some} \colorbox{red}{later} \textcolor{violet}{time} \textcolor{black}{we} \textcolor{brown}{think} \textcolor{black}{that} \textcolor{black}{the} \colorbox{cyan}{death} \textcolor{purple}{penalty} \textcolor{black}{>>*is*} \colorbox{cyan}{cruel} \textcolor{black}{or} \colorbox{cyan}{unusual,} \textcolor{black}{it} \textcolor{black}{will} \textcolor{black}{be} \colorbox{cyan}{outlawed.} \textcolor{black}{But} \textcolor{black}{at} \textcolor{black}{the} \textcolor{brown}{present,} \textcolor{black}{>>most} \textcolor{brown}{people} \textcolor{black}{don't} \textcolor{brown}{seem} \textcolor{black}{to} \textcolor{brown}{think} \textcolor{black}{this} \textcolor{brown}{way.} \textcolor{black}{>*This*} \textcolor{black}{from} \textcolor{black}{the} \textcolor{black}{same} \textcolor{brown}{fellow} \textcolor{black}{who} \textcolor{green}{speaks} \textcolor{black}{of} \textcolor{black}{an} \textcolor{brown}{"objective"} \textcolor{black}{or} \textcolor{brown}{"natural"} \textcolor{brown}{>morality.} \textcolor{black}{I} \textcolor{brown}{suppose} \textcolor{black}{that} \textcolor{black}{if} \textcolor{black}{the} \colorbox{cyan}{majority} \textcolor{brown}{decides} \textcolor{brown}{slavery} \textcolor{black}{is} \textcolor{black}{OK,} \textcolor{black}{then} \textcolor{black}{>it} \textcolor{black}{is} \textcolor{black}{no} \textcolor{teal}{longer} \textcolor{brown}{immoral?} \textcolor{black}{I} \textcolor{black}{did} \textcolor{black}{not} \textcolor{brown}{claim} \textcolor{black}{that} \textcolor{black}{our} \textcolor{brown}{system} \textcolor{black}{was} \textcolor{brown}{objective.} \textcolor{brown}{keith} \textcolor{black}{Subject:} \textcolor{black}{Re:} \colorbox{lime}{Space} \colorbox{lime}{Station} \colorbox{lime}{Redesign,} \colorbox{lime}{JSC} \textcolor{brown}{Alternative} \textcolor{black}{#4} \textcolor{black}{In} \textcolor{black}{<23APR199317452695@tm0006.lerc.nasa.gov>} \textcolor{black}{dbm0000@tm0006.lerc.nasa.gov} \textcolor{red}{(David} \textcolor{black}{B.} \textcolor{black}{Mckissock)} \textcolor{pink}{writes:} \colorbox{lime}{>Option} \textcolor{black}{"A"} \textcolor{black}{-} \colorbox{lime}{Low} \colorbox{lime}{Cost} \colorbox{lime}{Modular} \colorbox{lime}{Approach} \textcolor{black}{>} \textcolor{black}{-} \textcolor{brown}{Human} \textcolor{violet}{tended} \colorbox{lime}{capability} \textcolor{black}{(as} \colorbox{cyan}{opposed} \textcolor{black}{to} \textcolor{black}{the} \colorbox{gray}{old} \colorbox{lime}{SSF} \textcolor{black}{sexist} \textcolor{brown}{term} \textcolor{black}{>} \textcolor{black}{of} \textcolor{black}{man-tended} \colorbox{lime}{capability)} \colorbox{lime}{>Option} \textcolor{black}{"B"} \textcolor{black}{-} \colorbox{lime}{Space} \colorbox{lime}{Station} \colorbox{cyan}{Freedom} \colorbox{lime}{Derived} \textcolor{black}{>} \textcolor{black}{-} \textcolor{black}{Man-Tended} \colorbox{lime}{Capability} \colorbox{lime}{(Griffin} \textcolor{black}{has} \textcolor{black}{not} \textcolor{pink}{yet} \textcolor{red}{adopted} \textcolor{black}{non-sexist} \textcolor{black}{>} \textcolor{green}{language)} \colorbox{lime}{>Option} \textcolor{black}{C} \textcolor{black}{-} \colorbox{lime}{Single} \colorbox{lime}{Core} \colorbox{lime}{Launch} \colorbox{lime}{Station.} \textcolor{black}{I'll} \colorbox{cyan}{vote} \textcolor{black}{for} \textcolor{violet}{anything} \textcolor{black}{where} \textcolor{black}{they} \textcolor{black}{don't} \textcolor{violet}{feel} \colorbox{lime}{constrained} \textcolor{black}{to} \textcolor{red}{use} \textcolor{pink}{stupid} \textcolor{black}{and} \textcolor{violet}{ugly} \textcolor{black}{PC} \textcolor{brown}{phrases} \textcolor{black}{to} \colorbox{gray}{replace} \textcolor{green}{words} \textcolor{violet}{like} \textcolor{green}{'manned'.} \textcolor{black}{If} \textcolor{black}{they} \textcolor{violet}{think} \textcolor{black}{they} \textcolor{violet}{need} \textcolor{black}{to} \textcolor{black}{do} \textcolor{black}{that,} \textcolor{black}{they're} \textcolor{black}{more} \textcolor{black}{than} \textcolor{violet}{likely} \colorbox{cyan}{engaging} \textcolor{black}{in} \colorbox{cyan}{'politics} \textcolor{black}{and} \textcolor{red}{public} \colorbox{lime}{relations} \textcolor{black}{as} \textcolor{violet}{usual'} \textcolor{violet}{rather} \textcolor{black}{than} \colorbox{lime}{seriously} \textcolor{violet}{wanting} \textcolor{black}{to} \textcolor{pink}{actually} \textcolor{violet}{get} \textcolor{black}{into} \colorbox{lime}{space.} \textcolor{black}{So} \textcolor{black}{that} \textcolor{black}{eliminates} \colorbox{lime}{Option} \textcolor{black}{"A"} \textcolor{black}{from} \textcolor{black}{the} \textcolor{gray}{running.} \textcolor{black}{What} \textcolor{black}{do} \textcolor{black}{they} \colorbox{lime}{call} \textcolor{black}{a} \textcolor{green}{manned} \colorbox{lime}{station} \textcolor{black}{in} \colorbox{lime}{Option} \textcolor{black}{"C"?} \textcolor{black}{[I'm} \textcolor{pink}{actually} \textcolor{black}{about} \colorbox{lime}{half} \colorbox{lime}{serious} \textcolor{black}{about} \textcolor{black}{that.} \colorbox{cyan}{People} \textcolor{black}{should} \textcolor{black}{be} \textcolor{black}{more} \textcolor{blue}{concerned} \textcolor{black}{with} \textcolor{black}{grammatical} \colorbox{lime}{correctness} \textcolor{black}{and} \textcolor{pink}{actually} \textcolor{violet}{getting} \textcolor{black}{a} \textcolor{blue}{working} \colorbox{lime}{station} \textcolor{black}{than} \textcolor{black}{they} \textcolor{black}{are} \textcolor{black}{with} \colorbox{cyan}{'Political} \colorbox{lime}{Correctness'} \textcolor{black}{of} \textcolor{green}{terminology.]} \textcolor{black}{--} \textcolor{brown}{"Insisting} \textcolor{black}{on} \textcolor{violet}{perfect} \colorbox{cyan}{safety} \textcolor{black}{is} \textcolor{black}{for} \colorbox{cyan}{people} \textcolor{black}{who} \textcolor{black}{don't} \textcolor{black}{have} \textcolor{black}{the} \textcolor{gray}{balls} \textcolor{black}{to} \textcolor{green}{live} \textcolor{black}{in} \textcolor{black}{the} \textcolor{violet}{real} \colorbox{lime}{world."} \textcolor{black}{--} \textcolor{green}{Mary} \colorbox{lime}{Shafer,} \colorbox{lime}{NASA} \colorbox{lime}{Ames} \colorbox{lime}{Dryden} \textcolor{black}{------------------------------------------------------------------------------} \textcolor{black}{Fred.McCall@dseg.ti.com} \textcolor{black}{-} \textcolor{black}{I} \textcolor{black}{don't} \textcolor{green}{speak} \textcolor{black}{for} \textcolor{brown}{others} \textcolor{black}{and} \textcolor{black}{they} \textcolor{black}{don't} \textcolor{green}{speak} \textcolor{black}{for} \textcolor{black}{me.} \textcolor{black}{Subject:} \textcolor{black}{Re:} \textcolor{blue}{Insurance} \textcolor{black}{and} \textcolor{cyan}{lotsa} \textcolor{violet}{points...} \textcolor{black}{In} \textcolor{pink}{article} \textcolor{black}{<1993Apr19.152527.23658@iscnvx.lmsc.lockheed.com>} \textcolor{black}{jrlaf@sgi502.msd.lmsc.lockheed.com} \textcolor{black}{(J.} \textcolor{black}{R.} \textcolor{black}{Laferriere)} \textcolor{pink}{writes:} \textcolor{black}{|} \textcolor{violet}{|Now} \textcolor{violet}{now} \textcolor{pink}{Keith,} \textcolor{violet}{just} \colorbox{red}{calm} \textcolor{black}{down.} \textcolor{black}{What} \textcolor{black}{are} \textcolor{black}{you} \textcolor{black}{some} \textcolor{black}{prohibitionist} \textcolor{black}{prick?} \textcolor{black}{The} \textcolor{violet}{|point} \textcolor{black}{of} \textcolor{pink}{Andrew} \textcolor{black}{Infante's} \colorbox{green}{posting} \textcolor{black}{was} \textcolor{violet}{obvious} \textcolor{black}{to} \textcolor{black}{solicit} \textcolor{violet}{suggestions} \textcolor{red}{pertaining} \textcolor{black}{|to} \textcolor{black}{the} \textcolor{blue}{cost} \textcolor{black}{of} \textcolor{blue}{insurance} \textcolor{black}{and} \textcolor{black}{the} \textcolor{violet}{like.} \textcolor{black}{I} \textcolor{black}{don't} \textcolor{blue}{care} \textcolor{black}{if} \textcolor{black}{you} \textcolor{black}{are} \textcolor{black}{MADD} \textcolor{black}{or} \textcolor{black}{SADD} \textcolor{black}{or} \textcolor{violet}{|whatever;} \textcolor{cyan}{keep} \textcolor{black}{it} \textcolor{black}{to} \textcolor{black}{yourself,} \textcolor{black}{we'd} \textcolor{black}{all} \colorbox{green}{appreciate} \textcolor{black}{that.} \textcolor{violet}{Well,} \textcolor{violet}{simply} \textcolor{cyan}{put,} \textcolor{cyan}{drinking} \textcolor{black}{is} \textcolor{black}{irrelavent.} \textcolor{cyan}{Driving} \textcolor{cyan}{drunk} \textcolor{black}{is} \textcolor{black}{indefensable} \textcolor{black}{and} \textcolor{black}{unforgivable.} \textcolor{black}{There} \textcolor{black}{is} \textcolor{black}{a} \textcolor{cyan}{large} \textcolor{black}{differnece.} \textcolor{black}{But,} \textcolor{black}{then,} \textcolor{black}{with} \textcolor{black}{an} \textcolor{brown}{attitude} \textcolor{violet}{like} \textcolor{black}{yours,} \textcolor{black}{I} \textcolor{violet}{expect} \textcolor{black}{you'll} \textcolor{black}{be} \colorbox{red}{dead} \colorbox{red}{soon.} \textcolor{black}{I} \textcolor{violet}{just} \textcolor{violet}{hope} \textcolor{black}{you} \textcolor{black}{don't} \textcolor{cyan}{take} \textcolor{black}{a} \textcolor{brown}{human} \textcolor{black}{being} \textcolor{black}{out} \textcolor{black}{with} \textcolor{black}{you.} \textcolor{gray}{Dave} \textcolor{cyan}{Svoboda} \textcolor{black}{(svoboda@void.rtsg.mot.com)} \textcolor{black}{|} \textcolor{black}{"I'm} \textcolor{violet}{getting} \textcolor{cyan}{tired} \textcolor{black}{of} \textcolor{black}{90} \textcolor{cyan}{Concours} \textcolor{black}{1000} \textcolor{black}{(Mmmmmmmmmm!)} \textcolor{black}{|} \colorbox{red}{beating} \textcolor{black}{you} \textcolor{black}{up,} \textcolor{gray}{Dave.} \textcolor{black}{84} \textcolor{black}{RZ} \textcolor{black}{350} \textcolor{cyan}{(Ring} \textcolor{cyan}{Ding)} \textcolor{cyan}{(Woops!)} \textcolor{black}{|} \textcolor{black}{You} \textcolor{cyan}{never} \textcolor{violet}{learn."} \textcolor{cyan}{AMA} \textcolor{black}{583905} \textcolor{cyan}{DoD} \textcolor{black}{#0330} \textcolor{black}{COG} \textcolor{black}{939} \textcolor{purple}{(Chicago)} \textcolor{black}{|} \textcolor{black}{--} \textcolor{cyan}{Beth} \textcolor{black}{"Bruiser"} \textcolor{cyan}{Dixon}

``` r
for(doc_number in doc_number_list){

print(doc_number)

#find the unprocessed words in the selected document by splitting the document at white spaces

list_of_words_in_doc <- words(stripWhitespace(as.character(corpus[[doc_number]])), " ")

#first get a list with the word indices of the words in the document
temp_list = tolower(list_of_words_in_doc)
temp_list <- sapply(temp_list, function(x) if (x %in% stopwords()) NA else x)

index_list = words_to_indices[removePunctuation(temp_list,preserve_intra_word_dashes = TRUE)]
#replace NULLs by NAs
index_list <- lapply(index_list, function(x) if (is.null(x)) NA else x)

temp_z = z_frame[z_frame$rows == doc_number,]

#print out the latex code for the words in the document colored according to their most probable topic assignment


#sink(file = '../colortext/z_'+String(doc_number)+'.tex')
vect=c()
for(i in 1:length(list_of_words_in_doc)){
  index = index_list[[i]]
if(is.na(index)){
append(vect,cat("\\textcolor{",'black','}{',list_of_words_in_doc[[i]],'} ',sep=""))}
else {
  topic = temp_z[temp_z$cols == index & temp_z$accessed == 0,][1,"topic"]
  temp_z[temp_z$cols == index,][1,"accessed"] <- 1
  if(topic<=length(list_of_latex_colors1)){
    append(vect,cat("\\textcolor{",list_of_latex_colors1[topic],'}{',list_of_words_in_doc[[i]],'} ',sep=""))
  }

else {
append(vect,cat("\\colorbox{",list_of_latex_colors2[topic-length(list_of_latex_colors1)],'}{',list_of_words_in_doc[[i]],'} ',sep=""))}
}
}
#sink(file=NULL)

}
```

    ## [1] 6234
    ## \textcolor{black}{Subject:} \textcolor{black}{Re:} \textcolor{black}{?} \textcolor{black}{(was} \textcolor{black}{Re:} \colorbox{cyan}{"Cruel"} \textcolor{black}{(was} \textcolor{black}{Re:} \textcolor{teal}{<Political} \textcolor{brown}{Atheists?))} \textcolor{black}{sdoe@nmsu.edu} \textcolor{brown}{(Stephen} \textcolor{brown}{Doe)} \textcolor{pink}{writes:} \textcolor{black}{>>Of} \textcolor{brown}{course,} \textcolor{black}{if} \textcolor{black}{at} \textcolor{black}{some} \textcolor{green}{later} \colorbox{lime}{time} \textcolor{black}{we} \textcolor{green}{think} \textcolor{black}{that} \textcolor{black}{the} \colorbox{magenta}{death} \textcolor{purple}{penalty} \textcolor{black}{>>*is*} \colorbox{cyan}{cruel} \textcolor{black}{or} \colorbox{cyan}{unusual,} \textcolor{black}{it} \textcolor{black}{will} \textcolor{black}{be} \colorbox{magenta}{outlawed.} \textcolor{black}{But} \textcolor{black}{at} \textcolor{black}{the} \textcolor{blue}{present,} \textcolor{black}{>>most} \textcolor{teal}{people} \textcolor{black}{don't} \textcolor{brown}{seem} \textcolor{black}{to} \textcolor{pink}{think} \textcolor{black}{this} \textcolor{blue}{way.} \textcolor{black}{>*This*} \textcolor{black}{from} \textcolor{black}{the} \textcolor{black}{same} \textcolor{brown}{fellow} \textcolor{black}{who} \textcolor{green}{speaks} \textcolor{black}{of} \textcolor{black}{an} \textcolor{brown}{"objective"} \textcolor{black}{or} \textcolor{brown}{"natural"} \textcolor{brown}{>morality.} \textcolor{black}{I} \textcolor{green}{suppose} \textcolor{black}{that} \textcolor{black}{if} \textcolor{black}{the} \textcolor{gray}{majority} \colorbox{cyan}{decides} \textcolor{brown}{slavery} \textcolor{black}{is} \textcolor{black}{OK,} \textcolor{black}{then} \textcolor{black}{>it} \textcolor{black}{is} \textcolor{black}{no} \colorbox{lime}{longer} \textcolor{brown}{immoral?} \textcolor{black}{I} \textcolor{black}{did} \textcolor{black}{not} \textcolor{brown}{claim} \textcolor{black}{that} \textcolor{black}{our} \textcolor{blue}{system} \textcolor{black}{was} \textcolor{brown}{objective.} \textcolor{brown}{keith} [1] 17121
    ## \textcolor{black}{Subject:} \textcolor{black}{Re:} \colorbox{lime}{Space} \colorbox{lime}{Station} \colorbox{lime}{Redesign,} \colorbox{lime}{JSC} \textcolor{blue}{Alternative} \textcolor{black}{#4} \textcolor{black}{In} \textcolor{black}{<23APR199317452695@tm0006.lerc.nasa.gov>} \textcolor{black}{dbm0000@tm0006.lerc.nasa.gov} \colorbox{magenta}{(David} \textcolor{black}{B.} \textcolor{black}{Mckissock)} \textcolor{pink}{writes:} \colorbox{lime}{>Option} \textcolor{black}{"A"} \textcolor{black}{-} \colorbox{gray}{Low} \textcolor{blue}{Cost} \textcolor{red}{Modular} \textcolor{red}{Approach} \textcolor{black}{>} \textcolor{black}{-} \textcolor{brown}{Human} \textcolor{brown}{tended} \colorbox{lime}{capability} \textcolor{black}{(as} \textcolor{red}{opposed} \textcolor{black}{to} \textcolor{black}{the} \colorbox{lime}{old} \colorbox{lime}{SSF} \textcolor{black}{sexist} \textcolor{blue}{term} \textcolor{black}{>} \textcolor{black}{of} \textcolor{black}{man-tended} \colorbox{lime}{capability)} \colorbox{lime}{>Option} \textcolor{black}{"B"} \textcolor{black}{-} \colorbox{lime}{Space} \colorbox{lime}{Station} \colorbox{cyan}{Freedom} \colorbox{lime}{Derived} \textcolor{black}{>} \textcolor{black}{-} \textcolor{black}{Man-Tended} \colorbox{lime}{Capability} \colorbox{lime}{(Griffin} \textcolor{black}{has} \textcolor{black}{not} \textcolor{blue}{yet} \textcolor{green}{adopted} \textcolor{black}{non-sexist} \textcolor{black}{>} \textcolor{olive}{language)} \colorbox{lime}{>Option} \textcolor{black}{C} \textcolor{black}{-} \colorbox{lime}{Single} \colorbox{lime}{Core} \colorbox{lime}{Launch} \colorbox{lime}{Station.} \textcolor{black}{I'll} \textcolor{blue}{vote} \textcolor{black}{for} \textcolor{pink}{anything} \textcolor{black}{where} \textcolor{black}{they} \textcolor{black}{don't} \textcolor{violet}{feel} \colorbox{lime}{constrained} \textcolor{black}{to} \colorbox{lime}{use} \textcolor{pink}{stupid} \textcolor{black}{and} \textcolor{violet}{ugly} \textcolor{black}{PC} \textcolor{brown}{phrases} \textcolor{black}{to} \colorbox{teal}{replace} \textcolor{green}{words} \textcolor{red}{like} \colorbox{lime}{'manned'.} \textcolor{black}{If} \textcolor{black}{they} \textcolor{violet}{think} \textcolor{black}{they} \colorbox{gray}{need} \textcolor{black}{to} \textcolor{black}{do} \textcolor{black}{that,} \textcolor{black}{they're} \textcolor{black}{more} \textcolor{black}{than} \textcolor{violet}{likely} \colorbox{cyan}{engaging} \textcolor{black}{in} \colorbox{cyan}{'politics} \textcolor{black}{and} \textcolor{magenta}{public} \colorbox{lime}{relations} \textcolor{black}{as} \textcolor{violet}{usual'} \textcolor{blue}{rather} \textcolor{black}{than} \colorbox{lime}{seriously} \textcolor{violet}{wanting} \textcolor{black}{to} \textcolor{gray}{actually} \colorbox{green}{get} \textcolor{black}{into} \colorbox{lime}{space.} \textcolor{black}{So} \textcolor{black}{that} \textcolor{black}{eliminates} \colorbox{lime}{Option} \textcolor{black}{"A"} \textcolor{black}{from} \textcolor{black}{the} \textcolor{blue}{running.} \textcolor{black}{What} \textcolor{black}{do} \textcolor{black}{they} \textcolor{red}{call} \textcolor{black}{a} \colorbox{red}{manned} \colorbox{lime}{station} \textcolor{black}{in} \colorbox{lime}{Option} \textcolor{black}{"C"?} \textcolor{black}{[I'm} \textcolor{pink}{actually} \textcolor{black}{about} \textcolor{blue}{half} \textcolor{brown}{serious} \textcolor{black}{about} \textcolor{black}{that.} \textcolor{violet}{People} \textcolor{black}{should} \textcolor{black}{be} \textcolor{black}{more} \textcolor{red}{concerned} \textcolor{black}{with} \textcolor{black}{grammatical} \colorbox{lime}{correctness} \textcolor{black}{and} \textcolor{pink}{actually} \textcolor{violet}{getting} \textcolor{black}{a} \colorbox{lime}{working} \colorbox{lime}{station} \textcolor{black}{than} \textcolor{black}{they} \textcolor{black}{are} \textcolor{black}{with} \colorbox{cyan}{'Political} \colorbox{lime}{Correctness'} \textcolor{black}{of} \textcolor{magenta}{terminology.]} \textcolor{black}{--} \textcolor{brown}{"Insisting} \textcolor{black}{on} \colorbox{gray}{perfect} \colorbox{lime}{safety} \textcolor{black}{is} \textcolor{black}{for} \colorbox{cyan}{people} \textcolor{black}{who} \textcolor{black}{don't} \textcolor{black}{have} \textcolor{black}{the} \textcolor{gray}{balls} \textcolor{black}{to} \colorbox{red}{live} \textcolor{black}{in} \textcolor{black}{the} \textcolor{brown}{real} \textcolor{brown}{world."} \textcolor{black}{--} \textcolor{green}{Mary} \colorbox{lime}{Shafer,} \colorbox{lime}{NASA} \colorbox{lime}{Ames} \colorbox{lime}{Dryden} \textcolor{black}{------------------------------------------------------------------------------} \textcolor{black}{Fred.McCall@dseg.ti.com} \textcolor{black}{-} \textcolor{black}{I} \textcolor{black}{don't} \textcolor{green}{speak} \textcolor{black}{for} \textcolor{green}{others} \textcolor{black}{and} \textcolor{black}{they} \textcolor{black}{don't} \textcolor{green}{speak} \textcolor{black}{for} \textcolor{black}{me.} [1] 1729
    ## \textcolor{black}{Subject:} \textcolor{black}{Re:} \textcolor{blue}{Insurance} \textcolor{black}{and} \textcolor{cyan}{lotsa} \textcolor{violet}{points...} \textcolor{black}{In} \textcolor{cyan}{article} \textcolor{black}{<1993Apr19.152527.23658@iscnvx.lmsc.lockheed.com>} \textcolor{black}{jrlaf@sgi502.msd.lmsc.lockheed.com} \textcolor{black}{(J.} \textcolor{black}{R.} \textcolor{black}{Laferriere)} \textcolor{gray}{writes:} \textcolor{black}{|} \textcolor{blue}{|Now} \textcolor{violet}{now} \textcolor{pink}{Keith,} \textcolor{violet}{just} \colorbox{red}{calm} \textcolor{black}{down.} \textcolor{black}{What} \textcolor{black}{are} \textcolor{black}{you} \textcolor{black}{some} \textcolor{black}{prohibitionist} \textcolor{black}{prick?} \textcolor{black}{The} \textcolor{violet}{|point} \textcolor{black}{of} \textcolor{pink}{Andrew} \textcolor{black}{Infante's} \textcolor{pink}{posting} \textcolor{black}{was} \textcolor{violet}{obvious} \textcolor{black}{to} \textcolor{black}{solicit} \textcolor{violet}{suggestions} \textcolor{red}{pertaining} \textcolor{black}{|to} \textcolor{black}{the} \textcolor{blue}{cost} \textcolor{black}{of} \textcolor{blue}{insurance} \textcolor{black}{and} \textcolor{black}{the} \textcolor{cyan}{like.} \textcolor{black}{I} \textcolor{black}{don't} \textcolor{blue}{care} \textcolor{black}{if} \textcolor{black}{you} \textcolor{black}{are} \textcolor{black}{MADD} \textcolor{black}{or} \textcolor{black}{SADD} \textcolor{black}{or} \textcolor{violet}{|whatever;} \textcolor{violet}{keep} \textcolor{black}{it} \textcolor{black}{to} \textcolor{black}{yourself,} \textcolor{black}{we'd} \textcolor{black}{all} \colorbox{green}{appreciate} \textcolor{black}{that.} \textcolor{gray}{Well,} \textcolor{blue}{simply} \textcolor{violet}{put,} \textcolor{cyan}{drinking} \textcolor{black}{is} \textcolor{black}{irrelavent.} \textcolor{cyan}{Driving} \textcolor{cyan}{drunk} \textcolor{black}{is} \textcolor{black}{indefensable} \textcolor{black}{and} \textcolor{black}{unforgivable.} \textcolor{black}{There} \textcolor{black}{is} \textcolor{black}{a} \textcolor{gray}{large} \textcolor{black}{differnece.} \textcolor{black}{But,} \textcolor{black}{then,} \textcolor{black}{with} \textcolor{black}{an} \textcolor{teal}{attitude} \textcolor{cyan}{like} \textcolor{black}{yours,} \textcolor{black}{I} \colorbox{magenta}{expect} \textcolor{black}{you'll} \textcolor{black}{be} \colorbox{magenta}{dead} \colorbox{red}{soon.} \textcolor{black}{I} \textcolor{violet}{just} \textcolor{violet}{hope} \textcolor{black}{you} \textcolor{black}{don't} \textcolor{blue}{take} \textcolor{black}{a} \textcolor{brown}{human} \textcolor{black}{being} \textcolor{black}{out} \textcolor{black}{with} \textcolor{black}{you.} \textcolor{cyan}{Dave} \textcolor{cyan}{Svoboda} \textcolor{black}{(svoboda@void.rtsg.mot.com)} \textcolor{black}{|} \textcolor{black}{"I'm} \textcolor{cyan}{getting} \textcolor{cyan}{tired} \textcolor{black}{of} \textcolor{black}{90} \textcolor{cyan}{Concours} \textcolor{black}{1000} \textcolor{black}{(Mmmmmmmmmm!)} \textcolor{black}{|} \colorbox{red}{beating} \textcolor{black}{you} \textcolor{black}{up,} \textcolor{gray}{Dave.} \textcolor{black}{84} \textcolor{black}{RZ} \textcolor{black}{350} \textcolor{red}{(Ring} \textcolor{cyan}{Ding)} \textcolor{cyan}{(Woops!)} \textcolor{black}{|} \textcolor{black}{You} \textcolor{cyan}{never} \textcolor{blue}{learn."} \textcolor{cyan}{AMA} \textcolor{black}{583905} \textcolor{cyan}{DoD} \textcolor{black}{#0330} \textcolor{black}{COG} \textcolor{black}{939} \textcolor{purple}{(Chicago)} \textcolor{black}{|} \textcolor{black}{--} \textcolor{cyan}{Beth} \textcolor{black}{"Bruiser"} \textcolor{cyan}{Dixon}

``` r
for(doc_number in doc_number_list){

print(doc_number)

#find the unprocessed words in the selected document by splitting the document at white spaces

list_of_words_in_doc <- words(stripWhitespace(as.character(processedCorpus[[doc_number]])), " ")

#first get a list with the word indices of the words in the document


index_list = match(list_of_words_in_doc,topicModel_lda@terms,nomatch =NULL)
#replace NULLs by NAs
index_list <- lapply(index_list, function(x) if (is.null(x)) NA else x)

temp_z = z_frame[z_frame$rows == doc_number,]

#print out the latex code for the words in the document colored according to their most probable topic assignment


#sink(file = '../colortext/z_processed_'+String(doc_number)+'.tex')
vect=c()
for(i in 1:length(list_of_words_in_doc)){
  index = index_list[[i]]
if(is.na(index)){}
else {
  topic = temp_z[temp_z$cols == index & temp_z$accessed == 0,][1,"topic"]
  temp_z[temp_z$cols == index,][1,"accessed"] <- 1
  if(topic<=length(list_of_latex_colors1)){
    append(vect,cat("\\textcolor{",list_of_latex_colors1[topic],'}{',list_of_words_in_doc[[i]],'} ',sep=""))
  }

else {
append(vect,cat("\\colorbox{",list_of_latex_colors2[topic-length(list_of_latex_colors1)],'}{',list_of_words_in_doc[[i]],'} ',sep=""))}
}
}
#sink(file=NULL)
}
```

    ## [1] 6234
    ## \colorbox{cyan}{cruel} \textcolor{teal}{polit} \textcolor{brown}{atheist} \textcolor{brown}{stephen} \textcolor{brown}{doe} \textcolor{pink}{write} \textcolor{brown}{cours} \textcolor{green}{later} \colorbox{lime}{time} \textcolor{green}{think} \colorbox{magenta}{death} \textcolor{purple}{penalti} \colorbox{cyan}{cruel} \colorbox{cyan}{unusu} \colorbox{magenta}{outlaw} \textcolor{blue}{present} \textcolor{teal}{peopl} \textcolor{brown}{seem} \textcolor{pink}{think} \textcolor{blue}{way} \textcolor{brown}{fellow} \textcolor{green}{speak} \textcolor{brown}{object} \textcolor{brown}{natur} \textcolor{brown}{moral} \textcolor{green}{suppos} \textcolor{gray}{major} \colorbox{cyan}{decid} \textcolor{brown}{slaveri} \colorbox{lime}{longer} \textcolor{brown}{immor} \textcolor{brown}{claim} \textcolor{blue}{system} \textcolor{brown}{object} \textcolor{brown}{keith} [1] 17121
    ## \colorbox{lime}{space} \colorbox{lime}{station} \colorbox{lime}{redesign} \colorbox{lime}{jsc} \textcolor{blue}{altern} \colorbox{magenta}{david} \textcolor{pink}{write} \colorbox{lime}{option} \colorbox{gray}{low} \textcolor{blue}{cost} \textcolor{red}{modular} \textcolor{red}{approach} \textcolor{brown}{human} \textcolor{brown}{tend} \colorbox{lime}{capabl} \textcolor{red}{oppos} \colorbox{lime}{old} \colorbox{lime}{ssf} \textcolor{blue}{term} \colorbox{lime}{capabl} \colorbox{lime}{option} \colorbox{lime}{space} \colorbox{lime}{station} \colorbox{cyan}{freedom} \colorbox{lime}{deriv} \colorbox{lime}{capabl} \colorbox{lime}{griffin} \textcolor{blue}{yet} \textcolor{green}{adopt} \textcolor{olive}{languag} \colorbox{lime}{option} \colorbox{lime}{singl} \colorbox{lime}{core} \colorbox{lime}{launch} \colorbox{lime}{station} \textcolor{blue}{vote} \textcolor{pink}{anyth} \textcolor{violet}{feel} \colorbox{lime}{constrain} \colorbox{lime}{use} \textcolor{pink}{stupid} \textcolor{violet}{ugli} \textcolor{brown}{phrase} \colorbox{teal}{replac} \textcolor{green}{word} \textcolor{red}{like} \colorbox{lime}{man} \textcolor{violet}{think} \colorbox{gray}{need} \textcolor{violet}{like} \colorbox{cyan}{engag} \colorbox{cyan}{polit} \textcolor{magenta}{public} \colorbox{lime}{relat} \textcolor{violet}{usual} \textcolor{blue}{rather} \colorbox{lime}{serious} \textcolor{violet}{want} \textcolor{gray}{actual} \colorbox{green}{get} \colorbox{lime}{space} \colorbox{cyan}{elimin} \colorbox{lime}{option} \textcolor{blue}{run} \textcolor{red}{call} \colorbox{red}{man} \colorbox{lime}{station} \colorbox{lime}{option} \textcolor{pink}{actual} \textcolor{blue}{half} \textcolor{brown}{serious} \textcolor{violet}{peopl} \textcolor{red}{concern} \colorbox{lime}{correct} \textcolor{pink}{actual} \textcolor{violet}{get} \colorbox{lime}{work} \colorbox{lime}{station} \colorbox{cyan}{polit} \colorbox{lime}{correct} \textcolor{magenta}{terminolog} \textcolor{brown}{insist} \colorbox{gray}{perfect} \colorbox{lime}{safeti} \colorbox{cyan}{peopl} \textcolor{gray}{ball} \colorbox{red}{live} \textcolor{brown}{real} \textcolor{brown}{world} \textcolor{green}{mari} \colorbox{lime}{shafer} \colorbox{lime}{nasa} \colorbox{lime}{ame} \colorbox{lime}{dryden} \textcolor{green}{speak} \textcolor{green}{other} \textcolor{green}{speak} [1] 1729
    ## \textcolor{blue}{insur} \textcolor{cyan}{lotsa} \textcolor{violet}{point} \textcolor{cyan}{articl} \textcolor{gray}{write} \textcolor{blue}{now} \textcolor{violet}{now} \textcolor{pink}{keith} \textcolor{violet}{just} \colorbox{red}{calm} \textcolor{violet}{point} \textcolor{pink}{andrew} \textcolor{cyan}{infant} \textcolor{pink}{post} \textcolor{violet}{obvious} \colorbox{gray}{solicit} \textcolor{violet}{suggest} \textcolor{red}{pertain} \textcolor{blue}{cost} \textcolor{blue}{insur} \textcolor{cyan}{like} \textcolor{blue}{care} \textcolor{violet}{whatev} \textcolor{violet}{keep} \colorbox{green}{appreci} \textcolor{gray}{well} \textcolor{blue}{simpli} \textcolor{violet}{put} \textcolor{cyan}{drink} \textcolor{cyan}{drive} \textcolor{cyan}{drunk} \textcolor{gray}{larg} \textcolor{teal}{attitud} \textcolor{cyan}{like} \colorbox{magenta}{expect} \colorbox{magenta}{dead} \colorbox{red}{soon} \textcolor{violet}{just} \textcolor{violet}{hope} \textcolor{blue}{take} \textcolor{brown}{human} \textcolor{cyan}{dave} \textcolor{cyan}{svoboda} \textcolor{cyan}{get} \textcolor{cyan}{tire} \textcolor{cyan}{concour} \colorbox{red}{beat} \textcolor{gray}{dave} \textcolor{red}{ring} \textcolor{cyan}{ding} \textcolor{cyan}{woop} \textcolor{cyan}{never} \textcolor{blue}{learn} \textcolor{cyan}{ama} \textcolor{cyan}{dod} \textcolor{cyan}{cog} \textcolor{purple}{chicago} \textcolor{cyan}{beth} \textcolor{cyan}{dixon}

# LDAvis

Create a function to pass the correct arguments from the fit LDA model
to LDAvis::createJSON. This function is based on
<https://gist.github.com/trinker/477d7ae65ff6ca73cace> accessed on
\[01-11-2023\]

``` r
topicmodels2LDAvis <- function(x,DTM){
    post <- topicmodels::posterior(x)
    if (ncol(post[["topics"]]) < 3) stop("The model must contain > 2 topics")
    LDAvis::createJSON(
        phi = post[["terms"]],
        theta = post[["topics"]],
        vocab = colnames(post[["terms"]]),
        doc.length = slam::row_sums(DTM, na.rm = TRUE),
        term.frequency = slam::col_sums(DTM, na.rm = TRUE),
        R = 30
    )
}
```

Pass the model parameters to LDAvis to create an interactive
visualization.

``` r
LDAvis::serVis(topicmodels2LDAvis(topicModel_lda,DTM))
```

    ## Lade nötigen Namensraum: servr

\#Fit LSA model

convert the DTM to a quanteda DFM.

``` r
DFM = as.dfm(DTM)
DFM = dfm_weight(DFM,scheme='logcount')
```

Fit the LSA model with 150 topics

``` r
quanteda_lsa = quanteda.textmodels::textmodel_lsa(DFM,150)
```

\#LSAfun

money, buy, disk, car, health

``` r
LSAfun::plot_neighbors("buy",n=10,tvectors=quanteda_lsa$features,connect.lines='all')
```

    ##                    x           y            z
    ## buy        0.5342016 -0.49661628  0.469746190
    ## dealership 0.8330532 -0.24676525  0.174311713
    ## bought     0.5986347 -0.31201149  0.318956470
    ## price      0.4349906 -0.74213002  0.024818488
    ## market     0.1321676 -0.62796221  0.231981389
    ## purchas    0.1696317 -0.70968597  0.243962223
    ## dealer     0.7476712 -0.37104118 -0.077967454
    ## sell       0.2274021 -0.85450454 -0.007762237
    ## salesman   0.8115717 -0.08148702  0.183904592
    ## shop       0.1649560 -0.15124162  0.917206214

``` r
png("../bilder/buy_neighbours2d.png", width = 700*300/100,height = 500*300/100,res=300)
LSAfun::plot_neighbors("buy",n=10,tvectors=quanteda_lsa$features,connect.lines='all',dims=2)
```

    ##                    x           y
    ## buy        0.6635131 -0.52116883
    ## dealership 0.8311278 -0.25469659
    ## bought     0.6688516 -0.32831992
    ## price      0.3964082 -0.74161526
    ## market     0.1985122 -0.63971362
    ## purchas    0.2360741 -0.72191504
    ## dealer     0.6530779 -0.36483204
    ## sell       0.1900065 -0.85234806
    ## salesman   0.8182572 -0.09024909
    ## shop       0.5010310 -0.20195735

``` r
dev.off()
```

    ## png 
    ##   2

``` r
LSAfun::plot_wordlist(x = c("christian","atheism","buddhist","cathol","agnost","church",'jesus','god',"religion","pray","christ","belief","proof","judaism","jew","moral","ethic","islam","muslim"),connect.lines='all',tvectors = quanteda_lsa$features,dims = 3)
```

    ##                      x           y            z
    ## christian  0.392398273  0.63561497  0.260095328
    ## atheism    0.937371420 -0.08428078  0.080663275
    ## buddhist   0.496544052  0.35359252  0.195514442
    ## cathol     0.097767702  0.65454595  0.238131073
    ## agnost     0.864968684  0.05656932  0.146526525
    ## church    -0.025770162  0.73743962  0.119884141
    ## jesus     -0.133630167  0.87964841 -0.092587432
    ## god        0.484268212  0.66679251 -0.052350389
    ## religion   0.564234409  0.17718355  0.668114691
    ## pray       0.005888497  0.66406093 -0.270149886
    ## christ    -0.156327248  0.91929105 -0.078040262
    ## belief     0.740694213  0.23761179  0.221757173
    ## proof      0.663190746 -0.08010640 -0.150925103
    ## judaism    0.116459408  0.10346691  0.740593556
    ## jew       -0.163312858  0.11381666  0.720511081
    ## moral      0.359138114 -0.20328691  0.007963585
    ## ethic      0.116648861 -0.17214269  0.190741469
    ## islam      0.137584379 -0.11135066  0.771641840
    ## muslim     0.024227474 -0.06440096  0.711095191

``` r
LSAfun::plot_doclist(c("president talk budget","government discuss tax","study medic","doctor work in hospital","buy a car","purchas a vehicl","disk storage","i save a file"),tvectors=quanteda_lsa$features)#,dims=3)
```

    ## $coordinates
    ##                                         x           y            z
    ## document x1: president [...]  -0.06203548  0.74424796 -0.051053296
    ## document x2: government[...]   0.04234840  0.81796799  0.001753556
    ## document x3: study medi[...]  -0.48703184  0.12602907 -0.226735105
    ## document x4: doctor wor[...]  -0.30320990  0.15967615 -0.017625545
    ## document x5: buy a carNA[...]  0.73562301  0.14453319 -0.190203665
    ## document x6: purchas a [...]   0.81061455  0.13268168 -0.056777792
    ## document x7: disk stora[...]  -0.07607853  0.02535426  0.802861317
    ## document x8: i save a f[...]   0.05386737 -0.06479158  0.728626146
    ## 
    ## $xdocs
    ## [1] "document x1: president talk budget"  
    ## [2] "document x2: government discuss tax" 
    ## [3] "document x3: study medic"            
    ## [4] "document x4: doctor work in hospital"
    ## [5] "document x5: buy a car"              
    ## [6] "document x6: purchas a vehicl"       
    ## [7] "document x7: disk storage"           
    ## [8] "document x8: i save a file"

``` r
LSAfun::plot_neighbors('buy',n=9,tvectors=quanteda_lsa$features,dims=3)
```

    ##                    x           y          z
    ## buy        0.6092591 -0.37409504 0.44509758
    ## dealership 0.8424623 -0.18816007 0.20582326
    ## bought     0.6490543 -0.37081803 0.09409104
    ## price      0.4020065 -0.60398215 0.42603768
    ## market     0.1613489 -0.15466974 0.92103977
    ## purchas    0.1949250 -0.88516313 0.02453374
    ## dealer     0.6984544 -0.41582729 0.03022712
    ## sell       0.1860251 -0.72771430 0.43315815
    ## salesman   0.8256260 -0.01245007 0.17339514

``` r
LSAfun::plot_neighbors('buy',n=9,tvectors=quanteda_lsa$features,dims=2)
```

![](20Newsgroups-from-directory_files/figure-gfm/unnamed-chunk-34-1.png)<!-- -->

    ##                    x           y
    ## buy        0.6244058 -0.54625452
    ## dealership 0.8494929 -0.25114603
    ## bought     0.6563895 -0.34173435
    ## price      0.4202074 -0.72892888
    ## market     0.1839135 -0.64981613
    ## purchas    0.2084964 -0.73480957
    ## dealer     0.7050284 -0.34086301
    ## sell       0.2062905 -0.83978103
    ## salesman   0.8293464 -0.08897178
