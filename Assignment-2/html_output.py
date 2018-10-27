def generate_html(score_list,predicted,bucket):
    """
    generate html table
    :param score_list: the probability of classifying
    :return: html table
    """
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

    for i in range(4):
        html_line.append("<tr>\n")
        html_line.append("\t\t<th>"+classes[score_list[i]]+"</th>\n")
        for j in range(10):
            html_line.append("\t\t<th>" + str(round(bucket[i][j].item(),2)) + "</th>\n")
        html_line.append("\t\t<th>" + classes[predicted[i]] + "</th>\n")
        html_line.append("</tr>\n")

    html_line.append("</table>\n\n")
    html_line.append("</body>\n")
    html_line.append("</html>")
    return html_line

def write_lines(line_data_list, file_name):
    """
    write data to file
    :param line_data_list: data line list
    :param file_name: file_name
    :return: nothing
    """
    file_object = open(file_name, 'w')
    for line in line_data_list:
        file_object.write(line)
    file_object.close()