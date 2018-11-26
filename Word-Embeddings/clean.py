'''import re
punctuation=['0','1','2','3','4','5','6','7','8','9']
def isMallu(s):
	#print(s)
	for letter in s:
		#print(letter)
		if (u'\u0d00' <= letter <= u'\u0d7f')==False:
			#if letter not in punctuation :
			return False

	#print('Mallu'+s)
	return True

with open('v1.txt',encoding='utf8') as f_r:
	for line in f_r.readlines():
		print(line)	
		line = (bytes(line.strip(), 'utf-8').strip()).decode('utf-8', 'ignore')

		#dlist=data.split('\n')
		count=0
		newlist=[]
		if line!='':
			word_list=line.split()	
			#print(word_list)
			filtered_sen=[i for i in word_list if isMallu(i)==True]
			if len(filtered_sen)>1:
				sent=' '.join(filtered_sen)
				if sent not in newlist:
					print(sent)
					newlist.append(sent)

		while '' in newlist:
			newlist.remove('')


		with open('vyganews.txt','w',encoding='utf8') as f_w:
			for line in newlist:
				#print(line)
				f_w.write(line+'\n')
				'''
import re

def isMallu(s):
	#print(s)
	for letter in s:
		#print(letter)
		if (u'\u0d00' <= letter <= u'\u0d7f')==False:
			#if letter not in punctuation :
			return False

	#print('Mallu'+s)
	return True


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|[^\u0d00-\u0d7f .]*')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def cleantext(raw_html):
  cleanr = re.compile('[A-Za-z0-9,]*')
  clean = re.sub(cleanr, '', raw_html)
  return clean

with open('corpus/raw/Ganga2/gang_comb.txt',encoding='utf8') as f_r:
#with open('vyganews.txt',encoding='utf8') as f_r:
	raw=f_r.read()

clean=cleanhtml(raw)
rem_slash_n = re.compile('\n|\.\.+|  +|')
clean = re.sub(rem_slash_n, '', clean)


print('clean')
final=[]
lines=clean.split('.')
'''for line in lines:
	sentences=(line.strip()).split(' ')
	clean_sent=[]
	for word in sentences:
		if isMallu(word):
			clean_sent.append(word)
	#clean_sent.append('\n')
	
	final.append(' '.join(clean_sent))

while '' in final:
	final.remove('')'''

final = list(filter(None, lines)) # fastest
print('filtered')


#print(final)
with open('corpus/clean/gang_comb.txt','w',encoding='utf8') as f_w:
#with open('vygaclean.txt','w',encoding='utf8') as f_w:
	'''for line in final:
		f_w.write(line+'\n')'''
	f_w.write('\n'.join(final))
print('written')