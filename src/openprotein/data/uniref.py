from openprotein.core.config import DataConfig
from .dataset import DataFactory, Data


class Uniref(Data):
    """
    # this a bio dataset

    Args:
        path (str):path for the dataset

    Examples:
        Example1:
        >>> data = Uniref("./resources/uniref50/valid")
        >>> mydataset = data.get_data()
        >>> print(mydataset[0])
        MKWTNAGSRRGSKKAAPSARPLPVNLRLNDFSDDELHLATRRSTGNSPDAPPQAERVGYSQLTVLIAELRRSSRLGRSTCAEVTRHYPAIIYVFVFTRCLPQPNSCST

        Example2:
        >>> proteinseq_toks = {
                'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                        'X', 'B', 'U', 'Z', 'O', '.', '-']
            }
        >>> converter = MaskedConverter(proteinseq_toks["toks"])
        >>> f = lambda x: converter(x)
        >>> dl = data.get_dataloader(batch_size=4, collate_fn=f)
        >>> for i, j, k in dl
        >>> print(i)
        tensor([[32, 20, 15,  ..., 21, 11,  2],
                [32, 20,  8,  ...,  1,  1,  1],
                [32, 20,  8,  ...,  1,  1,  1],
                [32, 20, 18,  ...,  1,  1,  1]])
    """
    # super DataFactory, to device use which backend
    def __init__(self, path: str):
        # path = "./resources/uniref50/valid"
        super().__init__(path)

        # self._dataset = DataFactory.load(self, path)