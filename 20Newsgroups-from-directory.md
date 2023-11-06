20Newsgroups
================
Elisa Bankl
2023-10-01

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

Specify the location of the folder with the documents in the 20
Newsgroups dataset

``` r
directory_location = "C:/Users/elisa/Documents/VWA-2/20news-18828.tar/20news-18828/20news-18828"
```

``` r
corpus_source = DirSource(directory = directory_location,recursive = TRUE)
```

the newsgroup categories are in the names of the subfolders create a
vector with the newsgroups categories

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

remove the ‘’From:’ tags from the documents, to remove information that
might the newsgroups.

``` r
remove_from_tag <- content_transformer(function(x) {
  gsub("^From:.*", "", x)
})

corpus <- tm_map(corpus, remove_from_tag)
```

Shuffle corpus and metadata

only run once!!!!

``` r
set.seed(123)  # Set a random seed for reproducibility
shuffle_indices <- sample(length(corpus))
corpus <- corpus[shuffle_indices]
subfolder_names = subfolder_names[shuffle_indices]
```

\#Preprocessing corpus

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

compute another DTM with tfidf weighting

``` r
DTM_tfidf <- tm::DocumentTermMatrix(DTM, control = list(bounds = list(global = c(15, length(processedCorpus)*0.8)),weighting = weightTf(DTM)))
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

\#Fit LDA model

fit LDA using Gibbs Sampling

``` r
set.seed(523)
topicModel_lda <- topicmodels::LDA(DTM, 20, method="Gibbs", control=list(iter=2000,thin=2000,initialize = "random",best=TRUE))
```

\#Create heatmap

load the tidyr and the dplyr package

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

convert the matrix theta into a dataframe.

Try to mimic the way LDAvis determines the topic order.

``` r
topic_prevalence <- slam::row_sums(DTM, na.rm = TRUE)%*%(posterior(topicModel_lda)[["topics"]])
```

``` r
order = order(order(-topic_prevalence)) #order the topics by prevelance in the corpus
theta_frame= data.frame(topicModel_lda@gamma) #convert the matrix theta into a dataframe
topic_order = colnames(theta_frame)
names(topic_order) <- order #map the old document names to the number
```

Get a dataframe with the thetas in one column and the metadata in a
second column.

``` r
theta_wide <- tibble(theta_frame,'newsgroup'=unlist(subfolder_names),'doc_length'=slam::row_sums(DTM, na.rm = TRUE))

theta_wide %>% 
  select(!doc_length)%>%
  group_by(newsgroup) %>%
  summarise_if(is.numeric, mean, na.rm = TRUE)%>%
  tibble::remove_rownames() %>% 
  tibble::column_to_rownames(var="newsgroup")->theta_average
theta_average <- rename(theta_average, all_of(topic_order))
```

load pheatmap, RColorBrewer

``` r
library(pheatmap)
library(RColorBrewer)
```

Create heatmap and save it to png file.

``` r
png("../heatmap.png", width = 700*300/100,height = 500*300/100,res=300)
pheatmap(
  as.matrix(
      theta_average
    ), 
  color = colorRampPalette(brewer.pal(n = 7, name ="YlGnBu"))(100),
  border_color = NA,scale ='none',angle_col=0)
dev.off()
```

    ## png 
    ##   3

\#Print out documents with words colored according to their topic
assignment in z

Create a dtm with the unpreprocessed tokens, to get the unique tokens.

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

``` r
#list of colors that can be used to color text in latex
list_of_latex_colors1 = c("red","green","blue","cyan","magenta","gray","teal","violet",'lime','brown','purple','orange','pink','olive')
list_of_latex_colors2 = c("red","green","cyan","magenta","gray","teal","violet",'lime','brown','purple','orange','pink','olive')
```

Map the words to the indices of the processed word in the term list
returned by LDA.

``` r
words_to_indices <- match(unlist(word_mapping), topicModel_lda@terms, nomatch = NULL)
words_to_indices <- as.list(words_to_indices)
names(words_to_indices) <- names(word_mapping)
```

Create a dataframe with z and the corresponding document and word index

``` r
row_indices <- rep(DTM$i, DTM$v)
col_indices <- rep(DTM$j, DTM$v)
z_frame = data.frame(rows = row_indices,cols = col_indices,topic = topicModel_lda@z,accessed=0)
```

``` r
for(doc_number in c(231,6234,17121,15121)){

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

sink(file = '../colortext/'+String(doc_number)+'.tex')
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
sink(file=NULL)

}
```

``` r
for(doc_number in c(231,6234,17121,15121)){

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


sink(file = '../colortext/z_'+String(doc_number)+'.tex')
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
sink(file=NULL)

}
```

    ## [1] 231
    ## [1] 6234
    ## [1] 17121
    ## [1] 15121

``` r
for(doc_number in c(231,6234,17121,15121)){

print(doc_number)

#find the unprocessed words in the selected document by splitting the document at white spaces

list_of_words_in_doc <- words(stripWhitespace(as.character(processedCorpus[[doc_number]])), " ")

#first get a list with the word indices of the words in the document


index_list = match(list_of_words_in_doc,topicModel_lda@terms,nomatch =NULL)
#replace NULLs by NAs
index_list <- lapply(index_list, function(x) if (is.null(x)) NA else x)

temp_z = z_frame[z_frame$rows == doc_number,]

#print out the latex code for the words in the document colored according to their most probable topic assignment


sink(file = '../colortext/z_processed_'+String(doc_number)+'.tex')
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
sink(file=NULL)

}
```

    ## [1] 231
    ## [1] 6234
    ## [1] 17121
    ## [1] 15121

\#LDAvis

``` r
library(LDAvis)
```

Create a function to pass the correct arguments from the fit LDA model
to LDAvis::createJSON

``` r
# based on this code: https://gist.github.com/trinker/477d7ae65ff6ca73cace
#but use DTM instead of wordassignments for doc.length and term.frequency
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

``` r
LDAvis::serVis(topicmodels2LDAvis(topicModel_lda,DTM))
```

    ## Lade nötigen Namensraum: servr

\#Fit LSA model

convert the DTM to a quanteda DFM.

``` r
DFM = as.dfm(DTM)
DFM = dfm_tfidf(DFM)
```

``` r
quanteda_lsa = quanteda.textmodels::textmodel_lsa(DFM,150)
```

\#LSAfun

load the LSAfun package

``` r
library(LSAfun)
```

    ## Lade nötiges Paket: lsa

    ## Lade nötiges Paket: SnowballC

    ## Lade nötiges Paket: rgl

money, buy, disk, car, health

``` r
LSAfun::plot_neighbors("health",n=10,tvectors=quanteda_lsa$features,connect.lines='all')
```

    ##                  x         y          z
    ## health   0.3000714 0.9200663 -0.2504944
    ## care     0.3951121 0.2991171 -0.8655321
    ## souvien  0.9089679 0.2593033 -0.2990670
    ## gld      0.9088569 0.2592316 -0.2986467
    ## urgent   0.8935497 0.2525142 -0.3053810
    ## nyt      0.9195526 0.2611910 -0.2447458
    ## insur    0.9200803 0.2483221 -0.2646582
    ## provinci 0.8698879 0.2558405 -0.2578518
    ## hike     0.8281337 0.2101654 -0.3847765
    ## manitoba 0.8843270 0.2375109 -0.2446627

``` r
LSAfun::plot_wordlist(x = c("islam","judaism","christian","atheism","buddhist","cathol","agnost","church"),connect.lines='all',tvectors = quanteda_lsa$features,dims = 3)
```

    ##                      x            y           z
    ## islam     -0.025984637  0.025612204  0.46071920
    ## judaism   -0.159448263  0.059744214  0.43665336
    ## christian -0.186918533  0.028675044 -0.85696125
    ## atheism   -0.903059765 -0.022033401  0.22277230
    ## buddhist  -0.709485372 -0.061006177 -0.48216611
    ## cathol     0.004456337 -0.898639864  0.01499248
    ## agnost    -0.971220356 -0.002891595  0.06777410
    ## church    -0.050826010 -0.879900158 -0.10618757

``` r
LSAfun::plot_wordlist(x = c("car","bike","drive","oil","atheism","buddhist","cathol","agnost","church"),connect.lines='all',tvectors = quanteda_lsa$features,dims = 3)
```

    ##                     x            y             z
    ## car       0.015851232  0.027929678  0.7932032179
    ## bike     -0.019905987  0.011696180  0.6929709258
    ## drive     0.020527717 -0.008244857  0.4266158216
    ## oil       0.003342569 -0.008423487 -0.0008258228
    ## atheism  -0.911829533  0.086537425  0.0534403754
    ## buddhist -0.707793351 -0.087025284 -0.1157706023
    ## cathol   -0.062920349 -0.883320357  0.0231763972
    ## agnost   -0.974917027  0.084324555  0.0469080541
    ## church   -0.108052364 -0.887197356  0.0111396185

money buy

``` r
LSAfun::plot_neighbors(Predication("health","child",m=30,k=5,tvectors= quanteda_lsa$features)$PA,n=10,tvectors=quanteda_lsa$features,connect.lines='all')
```

    ##                      x         y         z
    ## Input Vector 0.2499528 0.9440189 0.1992344
    ## preval       0.5569264 0.2088798 0.7357759
    ## idaho        0.5517043 0.1475685 0.7577653
    ## outbreak     0.7071017 0.1677023 0.6603450
    ## strain       0.3236366 0.2231674 0.8580665
    ## tobacco      0.8587891 0.2191606 0.4487546
    ## diarrhea     0.8244852 0.1834425 0.5002179
    ## hamburg      0.8318129 0.1620276 0.4889252
    ## smoker       0.8539078 0.2724232 0.3190737
    ## ill          0.7873221 0.3025811 0.3793248

health money also: buy car buy money ethic religion

``` r
LSAfun::plot_neighbors(compose("vaccin","polit",tvectors=quanteda_lsa$features,method="Multiply"),n=9,tvectors=quanteda_lsa$features,connect.lines='all')
```

    ## Normalization does not change the orientation of result vector for this method

    ##                        x            y           z
    ## Input Vector  0.30100778 -0.114487039  0.63920961
    ## occup        -0.07654975  0.049639934  0.44171123
    ## villag        0.81865625  0.029386106  0.09401337
    ## herd          0.85742510 -0.134740974 -0.02375552
    ## workspac     -0.10875608  0.077607893  0.76742316
    ## fxwg          0.02280357 -0.984885536  0.04358710
    ## undesir       0.84811365  0.003774954 -0.04379744
    ## jubile        0.05362923 -0.988018084  0.03682831
    ## problemat     0.03428943 -0.080451677  0.30154398

health money

``` r
LSAfun::plot_neighbors(compose("car","ill",tvectors=quanteda_lsa$features,method="WeightAdd",a=1,b=3.5),n=9,tvectors=quanteda_lsa$features,connect.lines='all')
```

    ##                      x         y         z
    ## Input Vector 0.6730249 0.3923588 0.5858009
    ## mph          0.3393710 0.8751174 0.3426507
    ## sedan        0.4602824 0.5014485 0.7229361
    ## ford         0.8639835 0.2949283 0.3754697
    ## t-bird       0.7790468 0.3432301 0.4360596
    ## audi         0.5787835 0.4462018 0.6497528
    ## gts          0.8831731 0.3175575 0.3247069
    ## volvo        0.8445692 0.2746951 0.3866134
    ## tach         0.8689591 0.3363424 0.3110248

``` r
LSAfun::plot_neighbors(compose("food","diet",tvectors=quanteda_lsa$features,method="WeightAdd",a=1,b=0.5),n=9,tvectors=quanteda_lsa$features,connect.lines='all')
```

    ##                                        x          y         z
    ## Input Vector                   0.7633692 -0.3891977 0.4285491
    ## carcinogen                     0.5163607 -0.4695370 0.6493895
    ## corn                           0.1688656 -0.8670576 0.4002945
    ## potato                         0.8542525 -0.3324887 0.3436708
    ## eat                            0.4180345 -0.4128987 0.7722183
    ## imaharvardrayssdlinusmcspdccdy 0.9088405 -0.3010418 0.2191759
    ## sugar                          0.3668404 -0.7998943 0.1738452
    ## glutam                         0.9287212 -0.2510010 0.2445008
    ## food-rel                       0.3977393 -0.8000674 0.2596680
