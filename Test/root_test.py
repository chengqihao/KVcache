import stanza
#proxies = {'http': 'http://ip:port', 'https': 'http://ip:port'}
#stanza.download('en',proxies=proxies)
nlp = stanza.Pipeline(lang='en',processors='tokenize,mwt,pos,lemma,depparse')
doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
