import re
from openai import OpenAI
import base64
import requests
import time
import tqdm
import pandas as pd
import csv
import os
import pickle
import math

MAX_CBG_NUM = 15

PROMPT = """As a human mobility expert, your task is to analyze the patterns of visits by residents of various Census Block Groups (CBGs) to specific Points of Interest (POIs).
To accomplish this, you will assess the following data sets:
1. A concise summary of POI visitors' reviews and image descriptions for a specific destination POI, which may include the essence of the services offered by the POI, the sentiments and experiences of the visitors, consumer targeting, popularity, and any indications of age, religious, or racial preferences of the clientele.
2. A comprehensive set of features for part source CBGs within a 5km radius, including but not limited to distance from the POI, theoretical visit counts calculated through physical models, normalized average income levels, normalized total population, racial distribution, and educational attainment, formatted as a CSV file.
Your goal is to determine whether residents from different CBG sources will visit the POI. 
Let's think step by step.
1.Analyze the POI description from Dataset 1 to identify its unique features that may influence visitation patterns by residents of different source CBGs. Consider how the following attributes of the source CBGs might correlate with the likelihood of visitation to the POI: the distance from the POI, the total population size, average income levels, racial distribution, and educational attainment. 
For example: 1. Because of high consumer pricing at the POI, the source CBG's average income level is generally high; 2. Because the POI is a supermarket for daily necessities, which people generally prefer to be close to their homes, the source CBG's distance from POI is low; 3. Because the POI is a museum, attracting visitors regardless of distance, the source CBG's distance from POI is evenly distributed; 4. Because of POI offering services with racial or religious inclinations, the Non-Hispanic Black Ratio is high.
2.Based on the reasoning from step 1, and using the characteristics of the source CBGs from dataset 2, determine whether each of the CBGs is likely to have residents visiting the POI. Represent your findings with a list of boolean values (1 for yes, 0 for no), resulting in a list of {} elements.
Please respond in the following format：
' Reasons in Step one' : ' Your analysis', 'Result in Step two': ' [0,1,1,1,...,0,0,1,0,1]'
"""



def getresultlist(cbg_length, res):
    # 使用正则表达式匹配字符串中的列表
    match = re.search(r'\[(.*?)\]', res)
    if match:
        # 提取列表字符串
        list_str = match.group(1)
        try:
            # 将提取的字符串转换为实际的列表
            result_list = [int(item) for item in list_str.split(',')]
            # 检查列表长度和元素值
            if len(result_list) == cbg_length and all(item in [0, 1] for item in result_list):
                return result_list
            else:
                # 长度不匹配或元素值不是0或1，打印错误信息
                print("Error: List length must be {} and elements must be 0 or 1.".format(cbg_length))
        except ValueError:
            # 如果转换失败（例如列表中的元素不是整数），打印错误信息
            print("Error: The list contains non-integer values.")

    # 如果没有匹配的列表或出现错误，返回空列表
    return []


def mincbg(i,cbg_feature_flow_filt):

    total_count = cbg_feature_flow_filt.shape[0]
    start_index = i * MAX_CBG_NUM
    end_index = (i + 1) * MAX_CBG_NUM

    if total_count<=MAX_CBG_NUM:
        return total_count,cbg_feature_flow_filt
    else:
        if end_index > total_count:
            return total_count-start_index,cbg_feature_flow_filt.loc[start_index:,:]
        else:
            return MAX_CBG_NUM,cbg_feature_flow_filt.loc[start_index:(end_index-1),:]


def getInformation(i,poi_feature,cbg_feature_flow_filt):
    text_list = ['Here is the information:\n1. POI Summary.\n ', '2. CBG features and theoretical flow.\n']

    text_list[0] += poi_feature

    cbg_length,cbg_feature_flow_filt_limit = mincbg(i,cbg_feature_flow_filt)

    #将DataFrame转换为字符串
    cbg_df = cbg_feature_flow_filt_limit.drop(columns=['flow']).rename(columns={'predict_flow':'theoretical_visit_flow'})
    cbg_str = cbg_df.to_csv(index=False,sep=',')
    text_list[1] += cbg_str

    # 将text_list中的字符串拼接成一个完整的信息字符串
    Information = '\n'.join(text_list)
    return cbg_length, Information

# 定义预测函数
def predict(i,poi_feature,cbg_feature_flow_filt):
    start_time= time.time()

    cbg_length,info = getInformation(i,poi_feature,cbg_feature_flow_filt)
    print(i,cbg_length)

    retry_count = 100 #允许重试100次
    retry_interval = 1

    for _ in range(retry_count): #_占位符
        try:
            messages=[
                {
                    "role": "user",
                    "content": "You are a human mobility expert.\n"+PROMPT.format(cbg_length)+info
                }
            ]

            # print(payload["messages"])
            completion =  client.chat.completions.create(
                          model="Meta-Llama-3-8B-Instruct-AWQ",
                          messages=messages,
                          max_tokens=800,
                          temperature=1.1
                        )  #LM1 101.6.69.60 DL4 101.6.69.111

            msg = completion.choices[0].message.content
            # print(msg)
            print('total_tokens:', completion.usage)

            predict_list = getresultlist(cbg_length, msg)
            print("predict_list contains:", predict_list)

            end_time = time.time()
            total_run_time = round(end_time - start_time, 3)
            print('one_request_time: {} s'.format(total_run_time))
            return predict_list , msg

        except Exception as e:
            print("任务执行出错：", e)
            print('重新请求....')
            # retry_count += 1
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)



def main():
    start_time_total = time.time()

    cbg_feature_path = '/data2/fanbingbing/Segregation/data/Segregation/philadelphia/philadelphia_cbg_features_normalized_simplied.csv'
    POI_feature_path = '/data2/fanbingbing/Segregation/data/Segregation/philadelphia/philadelphia_poi_with_yelp_summary_GPT4v.csv'
    POI_flow_path = '/data2/fanbingbing/Segregation/data/baseline/philadelphia_all_ODpair_predictvsreal.csv'
    cbg_feature_columns = ['census_block_group','normalized_Total population','non_Hispanic_white_ratio',
                           'non_Hispanic_black_ratio','bachelor_ratio','normalized_average_income','unemployed_ratio']

    poi_df=pd.read_csv(POI_feature_path,usecols=['placekey','name','POI summary']).iloc[5000:,:] #取出index为a到b-1的行
    cbg_feature_df = pd.read_csv(cbg_feature_path,usecols=cbg_feature_columns) #1328*7
    POI_flow_df = pd.read_csv(POI_flow_path) #1625092*5

    predict_result = []
    placekey_reason_result = []
    for index,row in poi_df.iterrows():
        print(index)
        name = row['name']
        placekey = row['placekey']
        POI_flow_filt = POI_flow_df[POI_flow_df['placekey']==placekey].drop(columns=['placekey']).reset_index(drop=True)
        cbg_feature_flow_filt = POI_flow_filt.merge(cbg_feature_df,on='census_block_group',how='left')
        #按照flow大小排序，然后是distance
        # cbg_feature_flow_filt= cbg_feature_flow_filt.sort_values(by=['flow', 'distance'],
        #                                                                  ascending=[False,True]).reset_index(drop=True)

        positive_flow_count = (cbg_feature_flow_filt['flow'] > 0).sum()
        print(f"满足 flow > 0 的行数: {positive_flow_count}")
        total_count = cbg_feature_flow_filt.shape[0]
        if total_count % MAX_CBG_NUM == 1:
            cbg_feature_flow_filt = cbg_feature_flow_filt.iloc[:-1]
            total_count = total_count - 1
        print(f"总行数: {cbg_feature_flow_filt.shape[0]}")

        poi_result = []
        reason_result = []
        for i in range(math.ceil(total_count/MAX_CBG_NUM)):
            predict_list, msg= predict(i,row['POI summary'],cbg_feature_flow_filt)
            while not predict_list:
                predict_list,msg= predict(i,row['POI summary'],cbg_feature_flow_filt)
            else:
                poi_result.append(predict_list)
                reason_result.append(msg)

        flattened_list = [item for sublist in poi_result for item in sublist]
        predict_dict = {str(row['census_block_group'])[:-2]: flattened_list[idx] for idx, row in cbg_feature_flow_filt.iterrows()}
        predict_result.append([placekey,predict_dict])
        placekey_reason_result.append([placekey, reason_result]) #保存输出结果备用

        # 每20行保存一次text_result
        if (index+1) % 20 == 0:
            predict_result_df = pd.DataFrame(predict_result, columns=['placekey', 'predict_flow_01'])
            save_path = f'0511-8B-Instruction/judge_result_{index // 20}.csv'  # 计算文件名
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 如果目录不存在，则创建它
            predict_result_df.to_csv(save_path, index=False)
            del predict_result_df
            del predict_result
            predict_result = []

            reason_result_df = pd.DataFrame(placekey_reason_result, columns=['placekey', 'predict_reason'])
            save_path = f'0511-8B-Instruction-backup/judge_reason_{index // 20}.csv'  # 计算文件名
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 如果目录不存在，则创建它
            reason_result_df.to_csv(save_path, index=False)
            del reason_result_df
            del placekey_reason_result
            placekey_reason_result = []

    if len(predict_result) > 0:
        predict_result_df = pd.DataFrame(predict_result, columns=['placekey', 'predict_flow_01'])
        save_path = f'0511-8B-Instruction/judge_result_{index // 20}.csv'  # 计算文件名
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 如果目录不存在，则创建它
        predict_result_df.to_csv(save_path, index=False)
        del predict_result_df
        del predict_result
        predict_result = []

        reason_result_df = pd.DataFrame(placekey_reason_result, columns=['placekey', 'predict_reason'])
        save_path = f'0511-8B-Instruction-backup/judge_reason_{index // 20}.csv'  # 计算文件名
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 如果目录不存在，则创建它
        reason_result_df.to_csv(save_path, index=False)
        del reason_result_df
        del placekey_reason_result
        placekey_reason_result = []


    end_time_total = time.time()
    total_run_time_total = round(end_time_total-start_time_total, 3)
    print('Total_run_time: {} s'.format(total_run_time_total))


if __name__ == "__main__":
    api_base = "http://101.6.69.111:5000/v1"

    client = OpenAI(
            base_url="http://101.6.69.111:5000/v1/",
        api_key="token-fiblab-20240425",
    )

    main()
