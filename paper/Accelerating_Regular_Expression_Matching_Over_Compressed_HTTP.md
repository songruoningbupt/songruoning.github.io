### Accelerating_Regular_Expression_Matching_Over_Compressed_HTTP 压缩HTTP的快速正则解压匹配

#### INTRODUCTION

contributions

- A study of the challenges of regular expression matching over compressed traffic.
    - 压缩流量正则表达式匹配挑战的研究。
- A new method to allow the extraction of entire matched sub-strings for automata-based matching.
    - 一种允许提取完整匹配子字符串以实现自动机匹配的新方法。
- The first algorithmic framework that accelerates regular expression matching over compressed traffic.
    - 第一个算法框架，可加速压缩流量上的正则表达式匹配。
- Two architectural designs that achieve a significant performance improvement over traditional regular expression matching.
    - 与传统正则表达式匹配相比，两种架构设计可以显着提高性能。

#### BACKGROUND

- Compressed HTTP
    - GZIP compression first compresses the data using LZ77 compression and then encodes the literals and pointers using Huffman coding.
    - LZ77 算法 [LZ77压缩算法编码原理详解](https://www.cnblogs.com/junyuhuang/p/4138376.html)
    - Huffman coding哈夫曼编码
- Deep Packet Inspection Algorithms
- String Matching Over Compressed Traffic
    - LZ77 is an adaptive compression as each symbol is determined dynamically by the data. Therefore, there is no way to perform DPI without decompressing the data first
    - 在左边界和右边界的地方可能出问题，the DFA-based ACCH algorithm解决了这个问题
        - 算法过程
            - 扫描重复字符串的左边界并更新扫描结果。
            - 检查重复字符串的先前扫描结果是否包含匹配项。
            - 扫描重复字符串的右边框并更新扫描结果。
            - 更新重复字符串中跳过的字节的估计扫描结果。
        - 在ACCH中，先前字节的扫描结果存储在32K条目状态向量中。 其值由DFADepth（s）确定 - 从根状态到状态s的最短路径的长度。有3中可能的状态
            - MATCH：其中一个模式被匹配（匹配结束于被扫描的字节）。
            - UNCHECK：DFA-Depth（s）低于阈值参数CDepth（训练中，CDepth设置为2）。
            - CHECK：其他
        - ACCH将以以下方式使用Status-Vector
            - Left Border Scan：在处理重复的字符串时，ACCH扫描j个字节，直到达到j≥DFADepth（s）的状态s。 从这一点开始，任何模式前缀都已包含在重复的字符串区域中。 我们将重复字符串的前j个字节称为指针的左边界。

#### 名词

- traffic shaping: 流量整形
- content filtering: 内容过滤
