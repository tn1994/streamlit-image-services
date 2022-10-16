from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryModel:
    _query_idol_group_dict: dict = field(init=False)
    _query_kpop_group_dict: dict = field(init=False)
    _query_announcer_dict: dict = field(init=False)
    _query_actress_dict: dict = field(init=False)
    _query_cosplayers_dict: dict = field(init=False)

    query_category_dict: dict = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, '_query_idol_group_dict', {
            'モーニング娘。': ['鞘師里保', '小田さくら', '佐藤優樹', '工藤遥', '亀井絵里', '市井紗耶香', '石川梨華', ],
            'Berryz工房': ['清水佐紀', '嗣永桃子', '徳永千奈美', '須藤茉麻', '夏焼雅', '熊井友理奈', '菅谷梨沙子'],
            'S/mileage': ['S/mileage', '和田彩花', '前田憂佳', '福田花音', '小川紗季'],
            'Juice=Juice': ['宮崎由加', '金澤朋子', '高木紗友希', '大塚愛菜', '宮本佳林', '植村あかり'],
            'Hello! Project': ['真野恵里菜', '松浦亜弥', ],
            '欅坂46': ['欅坂46', '平手友梨奈', '志田愛佳', '渡邉理佐', '鈴本美愉', ],
            '日向坂46': ['日向坂46', '小坂菜緒', ],
            'Fairies': ['Fairies', '伊藤萌々香', '下村実生', '野元空', '林田真尋', '井上理香子', '清村川音', '藤田みりあ', ],
            'Other': ['新井愛瞳', '峮峮', ]
        })
        object.__setattr__(self, '_query_kpop_group_dict', {
            'IVE': ['IVE', '가을', '안유진', '레이', '장원영', '리즈', '이서', ],
            'NewJeans': ['NewJeans', '민지', '하니', '다니엘', '해린', '혜인'],
            'TWICE': ['TWICE', '나연', '지효', '정연', '모모', '사나', '미나', '다현', '채영', '쯔위', ],
            'CLC': ['CLC', '장승연', '오승희', '최유진', '장예은', '권은빈', '엘키', 'SORN'],
            'Kep1er': ['Kep1er', '최유진', '마시로', '샤오팅', '김채현', '김다연', '히카루', '휴닝바히에', '서영은', '강예서', ],
            '모모랜드': ['모모랜드', '혜빈', '제인', '나윤', '주이', '아인', '낸시', '연우', '태하', '데이지', ],
            'XG': ['XG', 'JURIN', 'CHISA', 'HARVEY', 'HINATA', 'JURIA', 'MAYA', 'COCONA'],

            'AOA': ['AOA', ],
            'Sister': ['Sister', ],
            'NiziU': ['NiziU', ],

        })
        object.__setattr__(self, '_query_announcer_dict', {
            '-': ['田中瞳', '檜山沙耶', ]
        })
        object.__setattr__(self, '_query_actress_dict', {
            '-': ['山田杏奈', '池間夏海', '山本美月', '石田ゆり子', '柴咲コウ', '大塚寧々', '章子怡', ]
        })
        object.__setattr__(self, '_query_cosplayers_dict', {
            '-': ['えなこ', '伊織もえ', '篠崎こころ', '十味', '火将ロシエル', ]
        })
        object.__setattr__(self, 'query_category_dict', {
            'Idol': self._query_idol_group_dict,
            'K-POP': self._query_kpop_group_dict,
            'Announcer': self._query_announcer_dict,
            'Actress': self._query_actress_dict,
            'Cosplayers': self._query_cosplayers_dict,
        })

    def get_query(self) -> dict:
        return self.query_category_dict
