from lhotse.recipes import mgb2

from bin.tokenizer import Tokenize

if __name__ == '__main__':
    try:
        mgb2.prepare_mgb2(corpus_dir="DataSet/mgb2_dev",output_dir="manifests",buck_walter=True)
    except:
        pass
    Tokenize("manifests","manifests/output","mgb2",['dev'])
