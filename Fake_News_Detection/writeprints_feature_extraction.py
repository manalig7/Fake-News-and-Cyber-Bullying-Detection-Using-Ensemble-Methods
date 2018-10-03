"""
CHARACTER BASED FEATURES
Total characters
Percentage of digits
Percentage of letters
Percentage of Uppercase letters
Percentage of whitespace
Percentage of tab spaces (NOT DOING)
Frequency of each letter (26)
Frequency of special characters (~ , @, #, $, %, ^, &, *, -, _, = ,+, >, <, [, ], {, }, /, \, |)

WORD BASED FEATURES
Total number of words
Percentage number of short words (less than 4 chracters)
Percentage of characters in words
Average word length
Average sentence length in terms of character
Average sentence length in terms of word
Total different words
Hapax legomena* Frequency of once-occurring words
Hapax dislegomena* Frequency of twice-occurring words
Yule’s K measure* A vocabulary richness measure defined by Yule
Simpson’s D measure* A vocabulary richness measure defined by Simpson
Sichel’s S measure* A vocabulary richness measure defined by Sichele
Brunet’s W measure* A vocabulary richness measure defined by Brune
Honore’s R measure* A vocabulary richness measure defined by Honore
Word length frequency distribution /Mnumber of words(20 features) 
Frequency of words in different length

SYNTACTIC FEATURES
Frequency of punctuations (8 features) “,”, “.”, “?”, “!”, “:”, “;”, “ ’ ” ,“ ” ”
Frequency of function words (303 features) The whole list of function words is in the appendix.

STRUCTURAL FEATURES
Total number of lines
Total number of sentences
Total number of paragraphs
Number of sentences per paragraph
Number of characters per paragraph
Number of words per paragraph
Has a greeting
Has separators between paragraphs
Has quoted content  -Cite original message as part of replying message
Position of quoted content -Quoted content is below or above the replying body
Indentation of paragraph -Has indentation before each paragraph
Use e-mail as signature
Use telephone as signature
Use url as signature

CONTENT SPECIFIC FEATURES
Frequency of content specific keyword
"""