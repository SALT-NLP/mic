from attribute_clf.common import *
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='label', help='the y label for the dataframe column to predict')
    parser.add_argument('--sent1', type=str, help='the label for the dataframe column to set as sentence 1')
    parser.add_argument('--sent2', type=str, help='the label for the dataframe column to set as sentence 2')
    parser.add_argument('--gpu', type=int, default=0, choices=list(range(8)))
    parser.add_argument('--input', type=str, default='splits', help='path to input file')
    parser.add_argument('--output', type=str, default='results', help='path to directory for outputting results')
    parser.add_argument('--maxlen', type=int, default=512, help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--save_model_every', type=int, default=0, help='save the model checkpoint every N epochs')
    parser.add_argument('--iters_to_accumulate', type=int, default=2, help='the gradient accumulation adds gradients over an effective batch of size : args.batch_size * iters_to_accumulate. If set to "1", you get the usual batch size')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--bert_model', type=str, default='albert-base-v2', choices=['albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', 'roberta-base'])
    parser.add_argument('--freeze_bert', action='store_true', help="if True, freeze the encoder weights and only update the classification layer weights")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    parser.add_argument('--backtranslate', type=str, default='')
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    df = pd.read_csv(args.input)

    df_train = df[df['split']=='train'].copy().reset_index()
    df_val = df[df['split']=='dev'].copy().reset_index()
    df_test = df[df['split']=='test'].copy().reset_index()
    del df
    
    if args.backtranslate != '':
        if args.sent2:
            raise ValueError('"backtranslate" argument is incompatible with a "sent2" argument')
        elif not os.path.exists(args.backtranslate):
            raise ValueError('Backtranslate file does not exist')
        else:
            print("Using backtranslations file", args.backtranslate)
            df_train = balance(df_train, balance_col=args.label, target=args.sent1, backtr_fn=args.backtranslate, random_state=args.seed).reset_index()
    
    if args.test:
        df_train = pd.concat([df_train, df_val]).reset_index(drop=True)
        df_val = df_test.copy()

    #  Set all seeds to make reproducible results
    set_seed(args.seed)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Creating instances of training and validation set
    print("Reading training data...")
    train_set = CustomDataset(df_train, args.maxlen, args.bert_model, sent1=args.sent1, sent2=args.sent2, lbl=args.label, eval_list_literal=False)
    print("Reading validation data...")
    val_set = CustomDataset(df_val, args.maxlen, args.bert_model, sent1=args.sent1, sent2=args.sent2, lbl=args.label, eval_list_literal=False)
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batchsize, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=args.batchsize, num_workers=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train = df_train[args.label].values
    #print('y_true', y_train)    

    criterion = nn.MSELoss()

    model = SentencePairClassifier(args.bert_model, freeze_bert=args.freeze_bert, labels_count=1)
    path_to_model = train_bert(model, args, criterion, df_train, df_val, train_loader, val_loader, save_model=args.test, is_regression=True) # only save if testing