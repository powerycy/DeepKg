# Copyright (c) 2021 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Chaoyi Yuan, chaoyiyuan3721@gmail.com
# Status: Active
import streamlit as st
from paddlenlp import Taskflow

st.set_page_config(page_title="CQUST-Intelligent Security Lab",page_icon=":使满意:")
st.title("UIE通用信息抽取框架FastDemo")
st.subheader("NER Part")
st.write("[参考文档](https://www.paddlepaddle.org.cn/tutorials/projectdetail/4035127)")
st.write("本Demo仅通过Paddle的Taskflow完成UIE任务,对于这种非常垂类的任务效果没有完全达到工业使用水平，因此需要一定的微调手段来完成UIE模型的微调来提升模型的效果")
# --- 用户输入 ---
example = st.selectbox(label="官方案例",options=["None","Example1","Example2","Example3"])

with st.container():
    with st.form(key="my_form"):
        st.write("以英文分号;为分隔单位")
        st.write("例如: 名称;毕业院校;职位;月收入;身体状况")
        if example == "Example1":
            example_schema = "名称;毕业院校;职位;月收入;身体状况"
            example_inputs = "兹证明凌霄为本单位职工，已连续在我单位工作5 年。学历为嘉利顿大学毕业，目前在我单位担任总经理助理  职位。近一年内该员工在我单位平均月收入（税后）为  12000 元。该职工身体状况良好。本单位仅此承诺上述表述是正确的，真实的。"
        elif example == "Example2":
            example_schema = "肿瘤部位;肿瘤大小"
            example_inputs = "胃印戒细胞癌，肿瘤主要位于胃窦体部，大小6*2cm，癌组织侵及胃壁浆膜层，并侵犯血管和神经。"
        elif example == "Example3":
            example_schema = "时间;赛手;赛事名称"
            example_inputs = "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"
        elif example == "None":
            example_schema = ""
            example_inputs = ""
        schema_inputs = st.text_input(label="输入模板",value=example_schema)
        schema = schema_inputs.split(";")
        inputs = st.text_area(label="输入抽取文字", height=120,value=example_inputs)
        inputs = inputs.replace("\n", "")[:512]
        submit_button = st.form_submit_button(label="✨ 启动!")

if len(schema_inputs)<=0:
    st.warning("请输入schema")
    st.stop()

if len(inputs)<=0:
    st.warning("请输入文字内容")
    st.stop()

if not submit_button:
    st.stop()

ie_model = Taskflow('information_extraction', schema=schema)
results = ie_model(inputs)

# --- 结果 ---
with st.container():
    st.write("---")
    with st.expander(label="json结果展示",expanded=False):
        st.write(results[0])
    for e,v in results[0].items():
        st.write("**{}**：{}".format(e,v[0]["text"]))