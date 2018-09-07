After executing the split params, meanings of generated file as follows:

...labels = The number here is representing the paragraphs in recall_paragraph file.
For example: Each question in test_question_5000_embeddings file has true paragraph in recall_paragraph_embeddings.hdf5 file.
So this match is happened due to the data in test_question_5000_labels.hdf5. This file is telling which question is assigned to which paragraph
in recall_paragraph_embeddings.hdf5.

...idx = This file is presenting the actual indx number of the questions stated in question_5000_embeddings file. So that
we can look up the paragraphs in actual squad datafile. For example. 1025 in test_question_5000_idx.hdf5 represent the question indx
in the actual questions.txt or object.