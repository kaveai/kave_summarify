# ka-ve_summarify

## Takım Üyeleri

- Yunus Emre Gündoğmuş
- Feyza Zeynep Salam
- Emre Yüksel
- Büşra Gökmen
- Hasan Kemik

## Kullanım

### Gerekli Kütüphaneler

```python
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import heapq
from gensim.summarization import keywords
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import networkx as nx
import re
import numpy as np
import json
import pickle
from keras.models import model_from_json
from keras.models import load_model
import Extraction_Based_Text_Summarization
```

Yukarıdaki kütüphaneleri programınıza ekledikten sonra,

```python
ex_sum = extraction_based_sum()
```
kodu ile class'ı çağırabilir. Sonrasında elinizde özetlenmesini istediğiniz veri ile

```python
ex_sum.get_sentences(text,k)
```

`k` kadar cümleyi metinden alabilirsiniz. Anahtar kelime çıkarımı için ise,

```python
ex_sum.get_keywords(text,ratio)
```
kodunu kullanarak, metindeki kelime sayısının oranı kadar anahtar kelime alabilirsiniz. Bu sayıyı `ratio` değeri ile kontrol edebilirsiniz.

## Referanslar
- [1] https://github.com/deeplearningturkiye/kelime_kok_ayirici
- [2] https://github.com/akoksal/Turkish-Word2Vec
- [3] https://github.com/Eruimdas/turkish_text_summarization
- [4] https://tscorpus.com


### ----------------------------------------------------------------------------------------------

## Team Members

- Yunus Emre Gündoğmuş
- Feyza Zeynep Salam
- Emre Yüksel
- Büşra Gökmen
- Hasan Kemik

## How to use

### Necessary Libraries

```python
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import heapq
from gensim.summarization import keywords
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import networkx as nx
import re
import numpy as np
import json
import pickle
from keras.models import model_from_json
from keras.models import load_model
import Extraction_Based_Text_Summarization
```

After you've added those libraries,

```python
ex_sum = extraction_based_sum()
```
you can use the class with that code, and after that, with the text you want to be summarized,

```python
ex_sum.get_sentences(text,k)
```

you can take the best `k` sentences from it. For keyword extraction,

```python
ex_sum.get_keywords(text,ratio)
```
with the code above, you can get keywords according to the ratio of the word count in the text. You can control that by changing `ratio` value.

## References
- [1] https://github.com/deeplearningturkiye/kelime_kok_ayirici
- [2] https://github.com/akoksal/Turkish-Word2Vec
- [3] https://github.com/Eruimdas/turkish_text_summarization
- [4] https://tscorpus.com

