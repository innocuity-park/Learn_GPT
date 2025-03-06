'''
本内容为自然语言处理任务之初探
主要学习实现文本向量化的任务
'''

from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt


tokenizer_gpt2 = AutoTokenizer.from_pretrained('gpt2')

text_fr = '''Évariste Galois (/ɡælˈwɑː/; français : [evaʁist ɡalwa] ; 25 octobre 1811 - 31 mai 1832) était un mathématicien français et un militant politique. Alors qu'il était encore adolescent, il parvint à déterminer une condition nécessaire et suffisante pour qu'un polynôme soit résoluble par des radicaux, résolvant ainsi un problème qui était resté ouvert pendant 350 ans. Son travail posa les fondements de la théorie de Galois et de la théorie des groupes, deux branches majeures de l'algèbre abstraite. Il était un fervent républicain et fut très impliqué dans les troubles politiques qui entourèrent la Révolution française de 1830. En raison de son activisme politique, il fut arrêté à plusieurs reprises, purgé une peine de plusieurs mois de prison. Pour des raisons restées obscures, peu de temps après sa libération de prison, il se battit en duel et décéda des blessures qu'il subit.'''
text_en = '''Évariste Galois (/ɡælˈwɑː/; French: [evaʁist ɡalwa]; 25 October 1811 – 31 May 1832) was a French mathematician and political activist. While still in his teens, he was able to determine a necessary and sufficient condition for a polynomial to be solvable by radicals, thereby solving a problem that had been open for 350 years. His work laid the foundations for Galois theory and group theory, two major branches of abstract algebra. He was a staunch republican and was heavily involved in the political turmoil that surrounded the French Revolution of 1830. As a result of his political activism, he was arrested repeatedly, serving one jail sentence of several months. For reasons that remain obscure, shortly after his release from prison he fought in a duel and died of the wounds he suffered.'''
text_zh = '''埃瓦里斯特·伽罗瓦（法语：Évariste Galois，1811年10月25日—1832年5月31日，法语发音： [evaʁist ɡalwa]）是一位法国数学家和政治活动家。尽管还在十几岁时，他就能够确定多项式能够通过根式求解的充分必要条件，从而解决了一个悬而未决的问题，该问题已经存在了350年。他的工作奠定了Galois理论和群论的基础，这两个是抽象代数的重要分支。他是一位坚定的共和派，深度参与了1830年法国大革命期间的政治动荡。由于他的政治活动，他多次被逮捕，其中一次入狱数月。由于原因不明，他在刑满释放后不久，参与了一场决斗并因受伤而去世。'''

texts = {
    'fr': text_fr,
    'en': text_en,
    'zh': text_zh
}

re = tokenizer_gpt2.encode(text_en) #得到一个数组，数组的每个元素来表示文字在字典里面的位置
#print(re)
#tokenizer_gpt2.decode(re) #将数组转换成文字

#比较一下分词前后的效果
def get_token_stats(tokenizer):
    str_stats = {}   #统计分词前的长度
    token_stats = {} #统计每个token出现的次数
    for (k, v) in texts.items():
        str_stats[k] = len(v.split()) if k != 'zh' else len(v) #split()默认是按空格分割
        token_stats[k] = len(tokenizer.encode(v))
    return str_stats, token_stats

#为了对中文预料有更好的分词效果，需要训练中文分词器
#分词器的训练是基于字典的，字典中的词越多，分词的效果越好
data = load_dataset('BellGroup/train_0.5M_CN')


#语言自卑与沙文主义