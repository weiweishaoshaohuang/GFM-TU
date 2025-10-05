import ast
import os
import re
import json

def split_checks(input_string):
    # pattern = r'[\w]+\(.*?\)'
    pattern = r'[\w]+\[.*?\]'
    # Use re.findall to get all matches
    result = re.findall(pattern, input_string)
    return result
def output_json_parser(input_string):
    input_string = input_string.strip().strip('\n').strip().replace('\n','')
    
    # 修复特殊情况：[]{"Function" : ...} 格式
    if input_string.startswith('[]'):
        input_string = input_string[2:].strip()
    
    json_match = re.search(r'```json(.+)```', input_string, re.DOTALL)
    json_match2 = re.search(r'\[.+\]', input_string)
    json_match3 = re.search(r'\{.+\}', input_string)

    match_str = []
    
    def safe_json_parse(json_str):
        """安全的JSON解析函数"""
        try:
            # 首先尝试json.loads
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # 如果失败，尝试ast.literal_eval
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                # 都失败就返回None
                return None
    
    if json_match:
        json_string = json_match.group(1).strip()
        match_str = safe_json_parse(json_string)
        if match_str is None:
            return [], []
    elif input_string.strip('```').strip('"').strip("'").strip().startswith('[') and input_string.strip(
            '```').strip('"').strip("'").strip().endswith(']'):
        cleaned_input = input_string.strip('```').strip('"').strip("'").strip()
        match_str = safe_json_parse(cleaned_input)
        if match_str is None:
            return [], []
    elif json_match2:
        json_str = json_match2.group()
        match_str = safe_json_parse(json_str)
        if match_str is None:
            return [], []
    
    if not match_str or (isinstance(match_str, list) and len(match_str) == 0) or (isinstance(match_str, list) and len(match_str) > 0 and (not isinstance(match_str[0], dict) or "Function" not in match_str[0])):
        if input_string.strip('```').strip('"').strip("'").strip().startswith('{') and input_string.strip(
                '```').strip('"').strip("'").strip().endswith('}'):
            cleaned_input = input_string.strip('```').strip('"').strip("'").strip()
            parsed = safe_json_parse(cleaned_input)
            if parsed is not None:
                match_str = [parsed]
            else:
                return [], []
        elif json_match3:
            json_str = json_match3.group()
            parsed = safe_json_parse(json_str)
            if parsed is not None:
                match_str = [parsed]
            else:
                return [], []
        else:
            return [],[]


    func_list = []
    exlanation_lsit = []
    for match in match_str:
        try:
            func_list.append(match['Function'])
            exlanation_lsit.append(
                f"Function: {match['Function']['function_name']}({', '.join([str(i) for i in match['Function']['parameters']])}), Explanation: {match['Explanation']}")
        except KeyError as k:
            print('LLM action 输出解析报错',k.__str__())
            raise Exception(f'LLM action 输出解析报错 {k.__str__()}')
            # continue
    return func_list, exlanation_lsit

def get_action_list(string):
    # if string[:len('Finish')] == 'Finish':
    #     return [string]
    # else:
    #     # return string.split(', ')
    # return split_checks(string)
    try:
        return output_json_parser(string)
    except Exception as e:
        print(f"JSON解析失败，原始输出：{string}")
        print(f"错误信息：{e}")
        # 如果解析失败，返回空的action列表，让程序继续运行
        return [], []



def parse_action_json(function_dict):
    action_type = function_dict['function_name']
    argument = function_dict['parameters']
    parameters = []
    for item in argument:
        if type(item) == list and len(item) > 0 and (type(item[0]) == tuple or (type(item[0]) == str and item[0].startswith('(') and item[0].endswith(')') and item[0].count(',') >= 2)):
            for it in item:
                parameters.append(ast.literal_eval(it) if type(it) == str and it.startswith('(') and it.endswith(')') and it.count(',') >= 2 else it )
        elif type(item) == list and 'answer' in (action_type.lower()):
            for it in item:
                parameters.append(it)
        elif type(item) == list:
            parameters.append(tuple(item))
        elif type(item) == str and item.startswith('(') and item.endswith(')') and item.count(',') >= 2:
            try:
                parameters.append(ast.literal_eval(item))
            except Exception as e:
                print('LLM输出action无法解析',item,e.__str__())
                # 尝试从复杂字符串中提取tuple
                try:
                    # 使用正则表达式提取所有tuple格式的内容
                    tuple_pattern = r'\(([^)]+)\)'
                    matches = re.findall(tuple_pattern, item)
                    if len(matches) >= 2:
                        # 如果找到多个tuple，取前两个作为参数
                        tuple1_parts = [part.strip().strip('"').strip("'") for part in matches[0].split(',')]
                        tuple2_parts = [part.strip().strip('"').strip("'") for part in matches[1].split(',')]
                        
                        # 构建两个tuple
                        tuple1 = (int(tuple1_parts[0]), int(tuple1_parts[1]), tuple1_parts[2])
                        tuple2 = (int(tuple2_parts[0]), int(tuple2_parts[1]), tuple2_parts[2])
                        
                        parameters.extend([tuple1, tuple2])
                    else:
                        # 如果只有一个tuple，按原来的逻辑处理
                        item_list = [i.strip().strip('(').strip(')').strip('"').strip("'").strip() for i in item.split(',')]
                        parameters.append((int(item_list[0]),int(item_list[1]),''.join(item_list[2:])))
                except Exception as parse_error:
                    print('复杂参数解析也失败',item,parse_error.__str__())
                    # 最后的兜底处理，直接传递原始字符串
                    parameters.append(item)
        elif type(item) == str and ',' in item and not item.startswith('('):
            # 处理形如 "4, 1, \"Available seat miles\"" 这样的字符串
            try:
                # 清理字符串并分割
                cleaned_item = item.strip().strip('"').strip("'")
                parts = [part.strip().strip('"').strip("'") for part in cleaned_item.split(',')]
                if len(parts) >= 3:
                    # 尝试构建tuple
                    row = int(parts[0])
                    col = int(parts[1])
                    content = ', '.join(parts[2:])  # 重新组合内容部分
                    parameters.append((row, col, content))
                else:
                    parameters.append(item)
            except Exception as e:
                print('字符串参数解析失败', item, e.__str__())
                parameters.append(item)
        else:
            # 处理字典格式的参数，如 {'node': "(3, 0, '2016')"}
            if isinstance(item, dict):
                # 如果是字典，尝试提取其中的值
                for key, value in item.items():
                    if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                        try:
                            parameters.append(ast.literal_eval(value))
                        except Exception:
                            # 如果ast.literal_eval失败，尝试手动解析
                            try:
                                cleaned_value = value.strip('()').strip()
                                parts = [part.strip().strip('"').strip("'") for part in cleaned_value.split(',')]
                                if len(parts) >= 3:
                                    row = int(parts[0])
                                    col = int(parts[1])
                                    content = ', '.join(parts[2:])
                                    parameters.append((row, col, content))
                                else:
                                    parameters.append(value)
                            except Exception:
                                parameters.append(value)
                    else:
                        parameters.append(value)
            else:
                parameters.append(item)
    return action_type,parameters

def LLM_json_output_parse(output):
    output = output.strip().strip('\n').strip().replace('\n','')
    json_match = re.search(r'```json(.+)```', output, re.DOTALL)
    # 使用正则表达式提取JSON
    json_pattern = r'\{.+\}'
    json_match2 = re.search(json_pattern, output)
    json_match3 = re.search(r'\[.+\]', output)
    if json_match:
        json_string = json_match.group(1)
        try:
            result = ast.literal_eval(json_string.strip().strip('\n'))
        except Exception as e:
            print('LLM输出解析报错', output, e.__str__())
            raise UserWarning('LLM输出解析报错 ' + output + e.__str__())
    elif output.strip('```').strip('"').strip("'").strip().startswith('{') and output.strip(
            '```').strip('"').strip("'").strip().endswith('}'):
        try:
            result = ast.literal_eval(output.strip('```').strip('"').strip("'").strip())
        except Exception as e:
            print('LLM输出解析报错', output, e.__str__())
            raise UserWarning('LLM输出解析报错 ' + output + e.__str__())
    elif output.strip('```').strip('"').strip("'").strip().startswith('[') and output.strip(
            '```').strip('"').strip("'").strip().endswith(']'):
        try:
            result = ast.literal_eval(output.strip('```').strip('"').strip("'").strip())
        except Exception as e:
            print('LLM输出解析报错', output, e.__str__())
            raise UserWarning('LLM输出解析报错 ' + output + e.__str__())
    elif json_match2:
        json_str = json_match2.group()
        # 将JSON字符串转换为字典
        result = ast.literal_eval(json_str)
    elif json_match3:
        json_str = json_match3.group()
        # 将JSON字符串转换为字典
        result = ast.literal_eval(json_str)
    else:
        raise UserWarning('LLM输出解析报错 ' + output)

    return result
def remove_quotes(s):
    s = s.strip().strip('\n')
    if s.startswith(("'", '"','‘','’','“','”')):
        s = s[1:]
    if s.endswith(("'", '"','‘','’','“','”')):
        s = s[:-1]
    return s



