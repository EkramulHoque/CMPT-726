def generate_html(test_images,predicted,probability,train_data,correct,total,precision):

    html = "<html>\n<body>\n\n<table border=>\n\t" \
           "<tr>\n" \
           "\t\t<th>Month</th>\n" \
           "\t\t<th>Savings</th>\n" \
           "\t</tr>\n" \
           "\t<tr>\n" \
           "\t\t<td>January</td>\n" \
           "\t\t<td>$100</td>\n" \
           "\t</tr>\n" \
           "</table>\n\n" \
           "</body>\n" \
           "</html>"

    html_line = []
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    html_line.append("<html>\n")
    html_line.append("<body>\n\n")
    html_line.append("<h1>Test Image and Its Classification Scores</h1>\n")
    html_line.append("<table border=>\n\t")

    html_line.append("<tr>\n")
    html_line.append("\t\t<th>Image</th>\n")
    html_line.append("\t\t<th> P(plane)</th>\n")
    html_line.append("\t\t<th> P(car)</th>\n")
    html_line.append("\t\t<th> P(bird)</th>\n")
    html_line.append("\t\t<th> P(cat)</th>\n")
    html_line.append("\t\t<th> P(deer)</th>\n")
    html_line.append("\t\t<th> P(dog)</th>\n")
    html_line.append("\t\t<th> P(frog)</th>\n")
    html_line.append("\t\t<th> P(horse)</th>\n")
    html_line.append("\t\t<th> P(ship)</th>\n")
    html_line.append("\t\t<th> P(truck)</th>\n")
    html_line.append("\t\t<th>The prediction</th>\n")
    html_line.append("\t</tr>\n")
    for i in range(2):
        for j in range(4):
            html_line.append("<tr>\n")
            html_line.append("\t\t<th>"+classes[test_images[i][j]]+"</th>\n")
            for k in range(len(classes)):
                html_line.append("\t\t<th>" + str(round(probability[i][j][k].item(),2)) + "</th>\n")
            html_line.append("\t\t<th>" + classes[predicted[i][j]] + "</th>\n")
            html_line.append("</tr>\n")
    html_line.append("</table>\n\n")

    html_line.append("<h1>Samples</h1>\n")
    html_line.append("<table border=>\n\t")
    html_line.append("<tr>\n")
    html_line.append("\t\t<th>Total Number of Testing Images</th>\n")
    html_line.append("\t\t<th>Total Number of Training Images</th>\n")
    html_line.append("\t</tr>\n")
    html_line.append("<tr>\n")
    html_line.append("\t\t<th>" + str(total) + "</th>\n")
    html_line.append("\t\t<th>" + str(train_data) + "</th>\n")
    html_line.append("\t</tr>\n")
    html_line.append("</table>\n\n")

    html_line.append("<h1>Testing Error of the network per Epoch</h1>\n")
    html_line.append("<table border=>\n\t")
    html_line.append("<tr>\n")
    html_line.append("\t\t<th>Epoch</th>\n")
    html_line.append("\t\t<th>Precision(%)</th>\n")
    html_line.append("\t</tr>\n")
    html_line.append("<tr>\n")
    for i in range(len(precision)):
        html_line.append("\t\t<th>" + str(i) + "</th>\n")
        html_line.append("\t\t<th>" + str(precision[i]+1) + "</th>\n")
        html_line.append("\t</tr>\n")
    html_line.append("</table>\n\n")

    html_line.append("</body>\n")
    html_line.append("</html>")
    return html_line

def write_lines(line_data_list, file_name):

    file_object = open(file_name, 'w')
    for line in line_data_list:
        file_object.write(line)
    file_object.close()