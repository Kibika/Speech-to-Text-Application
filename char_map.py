"""
Defines two dictionaries for converting 
between text and integer sequences.
"""


char_map_str = """
 ' 0
 <SPACE> 1
 a 2
 b 3
 c 4
 d 5
 e 6
 f 7
 g 8
 h 9
 i 10
 j 11
 k 12
 l 13
 m 14
 n 15
 o 16
 p 17
 q 18
 r 19
 s 20
 t 21
 u 22
 v 23
 w 24
 x 25
 y 26
 z 27
 N 15
 U 28
 K 29
 < 30
 > 31
 _ 32
 - 33
 . 34
 1 35
 2 36
 3 37
 4 38
 5 39
 6 40
 7 41
 8 42
 9 43
 ? 44
 0 45
 """

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch
index_map[2] = ' '