python ../Rethinking_Retrieve_Embeddings.py --window_length=4 --data_path="/media/jackalhan/Samsung_T5/Rethinking_Running_Data_QUASAR-T_short/dev/" --dataset_path="../dev.json" --conc_layers=-1,-2,-3,-4 --is_inject_idf=False --is_read_contents_from_squad_format=True --is_averaged_token=True --pre_generated_embeddings_path=bert_pre_trained_embeddings --embedding_type=bert --document_verbose=3 --max_tokens=-1 --is_stack_all=True