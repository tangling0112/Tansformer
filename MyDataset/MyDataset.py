from torch.utils.data import Dataset
from collections import Counter
import jieba


class MyDataset(Dataset):
    def __init__(self, path_en,path_zh, max_len, max_size, min_freq, training='train'):
        """
        :param path_en: 英语数据集的保存位置
        :param path_zh: 中文数据集的保存位置
        :param max_len: 句子的最大单词数
        :param max_size: 词典的最大容纳量
        :param min_freq: 词语出现的最小次数，少于该次数的词语将不会被记入词典
        :param training: 用于指定数据获取的方式
        """

        '''定义文本标识符'''
        pad_token = "<pad>"  # 用于填充字符串从而让每一个输入等长
        unk_token = "<unk>"  # 用于当我们的单词没有存在我们的词典中时，标志该单词为未知单词
        bos_token = "<bos>"  # 用于标识句子的开始
        eos_token = "<eos>"  # 用于标识句子的结束

        self.extra_tokens = [pad_token, unk_token, bos_token, eos_token]

        self.PAD = self.extra_tokens.index(pad_token)
        self.UNK = self.extra_tokens.index(unk_token)
        self.BOS = self.extra_tokens.index(bos_token)
        self.EOS = self.extra_tokens.index(eos_token)

        self.path_en = path_en
        self.path_zh = path_zh
        self.max_len = max_len
        self.max_size = max_size
        self.min_freq = min_freq

        '''读取文件获取数据集组成的二维向量'''
        self.examples_zh = self.read_zh_corpus(self.path_zh, self.max_len)
        self.examples_en = self.read_en_corpus(self.path_en, self.max_len)
        '''使用数据集的二维向量构建词典'''
        self.counter_en, self.word2idx_en, self.idx2word_en = self.build_En_vocab(
            self.examples_en, self.max_size, self.min_freq, self.extra_tokens)
        self.counter_zh, self.word2idx_zh, self.idx2word_zh = self.build_zh_vocab(
            self.examples_zh, self.max_size, self.min_freq, self.extra_tokens)
        '''使用词典将原数据集组成的二维向量转化成由字典索引组成的二维向量'''
        self.examples_en_idx = self.convert_text2idx(self.examples_en, self.word2idx_en)
        self.examples_zh_idx = self.convert_text2idx(self.examples_zh, self.word2idx_zh)
        self.examples_en_idx = self.add_padding(self.examples_en_idx,self.max_len)
        self.examples_zh_idx = self.add_padding(self.examples_zh_idx, self.max_len)
        #print(self.examples_zh_idx[1])
        self.vocab_size_zh = len(self.idx2word_zh)
        self.vocab_size_en = len(self.idx2word_en)

        if training == 'train':
            self.en_text_idx = self.examples_en_idx[0:6000]
            self.zh_text_idx = self.examples_zh_idx[0:6000]
        elif training == 'valid':
            self.en_text_idx = self.examples_en_idx[6001:8000]
            self.zh_text_idx = self.examples_zh_idx[6001:8000]
        else:
            self.en_text_idx = self.examples_en_idx[8001:10000]
            self.zh_text_idx = self.examples_zh_idx[8001:10000]

    def __len__(self):
        return len(self.en_text_idx)

    def __getitem__(self, idx):
        return self.en_text_idx[idx],self.zh_text_idx[idx]

    def get_vocab_size(self):
        return self.vocab_size_en, self.vocab_size_zh

    def build_zh_vocab(self, examples, max_size, min_freq, extra_tokens):
        """

        :param examples: 一个包含着中文数据集内所有文本的二维数组,每一行,为一句话
        :param max_size: 定义词典最大可包含的词语的数量
        :param min_freq: 用于指定一个词语的最小出现次数,如果小于该值,则这个词不会被添加到词典中
        :param extra_tokens: 一个存储着四个标识符的数组
        :return:我们的counter实例,单词为键,单词在idx2word中的索引为值的字典对象,保存了词典中所有单词的二维数组,每一行为一个单词
        """
        counter = Counter()  # 一个计数器实例,来自于Counter库
        word2idx, idx2word = {}, []
        '''将四个标识符添加到字典中'''
        if extra_tokens:
            idx2word += extra_tokens
            word2idx = {word: idx for idx, word in enumerate(extra_tokens)}

        min_freq = max(min_freq, 1)
        max_size = max_size + len(idx2word) if max_size else None  # 由于扩展到四个标识符不算到词典最大长度中,因此将max_size扩大
        for sent in examples:  # 迭代examples数组的每一行,也即数据集的每一句话.
            for w in sent:  # 迭代sent数组的每一个元素,也即一句话中的每一个单词
                counter.update([w])
        sorted_counter = sorted(counter.items(), key=lambda tup: tup[0])  # 按照单词词序进行排列
        '''
        每一个counter的item都是一个含有两个元素的数组,第一个元素指的是被添加到counter实例的单词,第二个为其出现的次数
        '''
        sorted_counter.sort(key=lambda tup: tup[1], reverse=True)  # 再按照单词出现的频率从大到小排序

        for word, freq in sorted_counter:
            if freq < min_freq or (max_size and len(idx2word) == max_size):
                break

            idx2word.append(word)  # 将单词添加到idx2word数组中
            word2idx[word] = len(idx2word) - 1
            '''
            1.idx2word数组中添加一个单词,此时长度为n+1
            2.在word2idx字典中以单词字符串为键,该单词在idx2word中的索引值为值添加
            '''
        return counter, word2idx, idx2word

    def build_En_vocab(self, examples, max_size, min_freq, extra_tokens):
        """

        :param examples: 一个包含着英文数据集内所有文本的二维数组,每一行,为一句话
        :param max_size: 定义词典最大可包含的单词的数量
        :param min_freq: 用于指定一个单词的最小出现次数,如果小于该值,则这个词不会被添加到字典中
        :param extra_tokens: 一个存储着四个标识符的数组
        :return: 我们的counter实例,单词为键,单词在idx2word中的索引为值的字典对象,保存了词典中所有单词的二维数组,每一行为一个单词
        """
        counter = Counter()  # 一个计数器实例,来自于Counter库
        word2idx, idx2word = {}, []
        '''将四个标识符添加到字典中'''
        if extra_tokens:
            idx2word += extra_tokens
            word2idx = {word: idx for idx, word in enumerate(extra_tokens)}

        min_freq = max(min_freq, 1)
        max_size = max_size + len(idx2word) if max_size else None  # 由于扩展到四个标识符不算到词典最大长度中,因此将max_size扩大
        for sent in examples:  # 迭代examples数组的每一行,也即数据集的每一句话.
            for w in sent:  # 迭代sent数组的每一个元素,也即一句话中的每一个单词
                counter.update([w])
        sorted_counter = sorted(counter.items(), key=lambda tup: tup[0])  # 按照单词词序进行排列
        '''
        每一个counter的item都是一个含有两个元素的数组,第一个元素指的是被添加到counter实例的单词,第二个为其出现的次数
        '''
        sorted_counter.sort(key=lambda tup: tup[1], reverse=True)  # 再按照单词出现的频率从大到小排序

        for word, freq in sorted_counter:
            if freq < min_freq or (max_size and len(idx2word) == max_size):
                break

            idx2word.append(word)  # 将单词添加到idx2word数组中
            word2idx[word] = len(idx2word) - 1
            '''
            1.idx2word数组中添加一个单词,此时长度为n+1
            2.在word2idx字典中以单词字符串为键,该单词在idx2word中的索引值为值添加
            '''
        return counter, word2idx, idx2word

    def read_zh_corpus(self, src_path, max_len):
        """
        :param src_path: 中文数据集保存的文件地址
        :param max_len: 读取的一句话可以包含的最多单词数
        :return: 中文数据集的二维数组
        """
        src_sents = []
        empty_lines, exceed_lines = 0, 0
        with open(src_path,encoding='utf8') as src_file:
            for idx, src_line in enumerate(src_file):
                if idx == 10000:
                    break
                if src_line.strip() == '':  # 当遇到空行时，让empty_lines加1并直接跳过这一行
                    empty_lines += 1
                    continue
                src_words = jieba.lcut(src_line.strip())  # 使用jieba库的精确搜索模式对句子进行词语拆分
                if max_len is not None and len(src_words) > max_len:  # 判断指定行包含的单词数量是否超过max_len如果超过则跳过该行，不将其录入
                    exceed_lines += 1
                    continue
                src_sents.append(src_words)  # 添加读取到的行到src_sents数组中形成二维数组
        return src_sents

    def read_en_corpus(self, src_path, max_len, lower_case=False):
        """
        :param src_path: 英文数据集保存的文件地址
        :param max_len: 读取的一句话可以包含的最多单词数
        :param lower_case: 用于指定是否将文本转化为小写
        :return: 英文数据集的二维数组
        """
        src_sents = []
        empty_lines, exceed_lines = 0, 0
        with open(src_path,encoding='utf8') as src_file:
            for idx, src_line in enumerate(src_file):
                if idx == 10000:
                    break
                if src_line.strip() == '':  # 当遇到空行时，让empty_lines加1并直接跳过这一行
                    empty_lines += 1
                    continue
                if lower_case:  # 用于对文本进行小写转换
                    src_line = src_line.lower()

                src_words = src_line.strip().split()
                if max_len is not None and len(src_words) > max_len:  # 判断指定行包含的单词数量是否超过max_len如果超过则跳过该行，不将其录入
                    exceed_lines += 1
                    continue
                src_sents.append(src_words)  # 添加读取到的行到src_sents数组中形成二维数组
        return src_sents

    def add_padding(self, idx_text, max_len):
        """

        :param idx_text: 已经转化为词典索引标识的代表数据集的二维数组,每一行表示一个句子
        :param max_len: 指示最大句子包含词语数,如果小于该值,则添加一定个数PAD符号填充到max_len长度
        :return: 添加了PAD之后的数据集的索引形式二维数组
        """
        PADadded_idx_text = []
        for idx_line in idx_text:
            if len(idx_line) == max_len:
                continue
            elif len(idx_line) < max_len:
                for i in range(max_len - len(idx_line)):
                    idx_line.append(self.PAD)
            PADadded_idx_text.append(idx_line)
        return PADadded_idx_text  # 已经添加了PAD的代表数据集的二维索引值数组

    def convert_idx2en_text(self, example, idx2word):
        """
        :param example: 一个有单词索引组成的一维数组
        :param idx2word:
        :return: 一个字符串
        """
        words = []
        for i in example:
            if i == self.EOS:  # 当遇到结束标识符时终止
                break
            if i == self.BOS:
                continue
            words.append(idx2word[i])  # 将该索引对应的单词
        return ' '.join(words)  # 将单词数组以每个元素之间相隔一个空白符的规则转化为字符串

    def convert_idx2zh_text(self, example, idx2word):
        pass

    def convert_text2idx(self, examples, word2idx):
        """

        :param examples: 一个容纳了数据集内所有句子的二维数组,每一行为一个句子
        :param word2idx: 数据集的索引形式表示的二维数组
        :return:
        """
        idx_text = []  # 用于容纳整个文本数据集的用词典索引替代单词的二维数组,每一行为数据集中的一句话
        idx_line = []  # 用于容纳单个句子的用词典索引代替单词的一维数组
        for sent in examples:
            idx_line.append(self.BOS)  # 在句子首部添加BOS开始符号
            for w in sent:
                if w in word2idx:
                    idx_line.append(word2idx[w])
                else:
                    idx_line.append(self.UNK)
            idx_line.append(self.EOS)  # 在句子尾部添加EOS句子结束符号
            idx_text.append(idx_line)
            idx_line = []
        return idx_text
