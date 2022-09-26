import os
import imageio

color_label_file = "CamVid/label_colors.txt"

def idimage2colorimage(image_id):
    label2color = {}
    
    classid2label = [" "]  # 用索引作为class_id对应label，因为class_id从1开始，所以用' '占用索引为0的元素
    with open(color_label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.split()[-1]
            color = [int(x) for x in list(line.split()[:-1])]
            label2color[label] = color
            classid2label.append(label)


    image_color = []
    for line_id in range(image_id.shape[0]):
        color_line = []
        for id in range(image_id.shape[1]):
            pixel = image_id[line_id][id]
            try:
                pixel_color = label2color[classid2label[pixel]]
            except:
                print(pixel)
            color_line.append(pixel_color)
        image_color.append(color_line)

    return image_color

def main():
    image_id = imageio.imread(
        "../CamVid/camvid/labels/0006R0_f01260_P.png")
    image_color = idimage2colorimage(image_id)
    imageio.imwrite("example.png", image_color)


if __name__ == "__main__":
    main()