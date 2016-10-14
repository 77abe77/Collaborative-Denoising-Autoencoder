from datasets.CiteYouLikeA import CiteYouLikeA 
from models.CDAE import CDAE

model = CDAE(CiteYouLikeA, sparse=False)
model.train()




