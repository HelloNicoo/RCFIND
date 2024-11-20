# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 20:07
# @Author  : sylviazz
# @FileName: labels

entity_type = ['药物', '检查', '症状', '流行病学', '疾病', '社会学', '预后', '其他治疗', '其他', '部位', '手术治疗']

rel_type = {'实验室检查', '病理分型', '外侵部位', '药物治疗', '预后生存率', '相关（症状）', '同义词', '多发地区', '相关（导致）', '发病部位', '传播途径', '多发季节', '预后状况', '手术治疗', '放射治疗', '影像学检查', '病理生理', '临床表现', '组织学检查', '病史', '辅助检查', '化疗', '预防', '病因', '侵及周围组织转移的症状', '并发症', '发病率', '遗传因素', '死亡率', '内窥镜检查', '多发群体', '鉴别诊断', '治疗后症状', '风险评估因素', '阶段', '发病性别倾向', '相关（转化）', '转移部位', '辅助治疗', '发病机制', '筛查', '发病年龄', '高危因素', '就诊科室', '无关系'}

def get_entity_category(task: str, data_type: str):
    if task.lower() == 'ner':
        categorys = entity_type
    elif task.lower() == 're':
        categorys = rel_type
    entity2id = {c: i for i, c in enumerate(categorys)}
    id2entity = {i: c for i, c in enumerate(categorys)}
    return [categorys] + [entity2id] + [id2entity]

def get_rel_type():
    rel2id = {s: i for i, s in enumerate(rel_type)}
    id2rel = {i: s for i, s in enumerate(rel_type)}
    return [rel_type] + [rel2id] + [id2rel]