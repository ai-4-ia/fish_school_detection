import os


# # Create a hardcoded dictionary which map the current classid to classid agreed by Yolo
# # Some classes which do not have any training and validating data will be moved out
# # For example:
# classid_detectron_yolo = {
#     '1': '0',
#     '2': '1',
#     '3': '2',
#     '6': '3',
#     '7': '4',
#     '8': '5',
#     '9': '6',
#     '11': '7',
#     '12': '8',
#     '13': '9',
#     '17': '10',
#     '19': '11',
#     '99': '12',
#     '100': '13'
# }
def map_to_yolo_classid(classid_detectron_yolo, folder_path):
    for yolo_txt_file in os.listdir(folder_path):
        data = open(os.path.join(folder_path, yolo_txt_file), "r+")

        updated_rows = []
        rows = [line.strip() for line in data.readlines()]
        for row in rows:
            row_token = row.split(' ')
            row_token[0] = classid_detectron_yolo[row_token[0]]
            updated_rows.append(' '.join(row_token) + '\n')
        data.flush()
        data.seek(0)
        data.writelines(updated_rows)
        data.truncate()
        data.close()
