import sys
sys.path.append('..//')
from extend.email import Email as Em
mail=Em()
mail.send()
del mail