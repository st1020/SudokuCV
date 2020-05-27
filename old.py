
    # 模板匹配
    #cross = np.zeros((128, 128), np.uint8)
    #cv2.rectangle(cross, (0, 60), (128, 68), 255, cv2.FILLED)
    #cv2.rectangle(cross, (60, 0), (68, 128), 255, cv2.FILLED)
    # 根据九宫格大小绘制近似的形状
    # x = int(np.sqrt(cnt[1])/27)
    # if x%2 == 1:
        # x += 1
    # x1 = int(x/2-1)
    # x2 = int(x/2+1)
    # print(x, x1, x2)
    # cross = np.zeros((x, x), np.uint8)
    # cv2.rectangle(cross, (0, x1), (x, x2), 255, cv2.FILLED)
    # cv2.rectangle(cross, (x1, 0), (x2, x), 255, cv2.FILLED)
    # 
    # show.add(cross, 'cross')
    # 
    # cross_contour, _ = cv2.findContours(cross, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(img_with_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 
    # img_contours = cv2.cvtColor(img_with_mask, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite('mask_contours.png', img_contours)
    # 
    # res = cv2.matchTemplate(img_with_mask, cross, cv2.TM_CCOEFF)
    # cv2.imwrite('res.png', res)
    # 
    # 
    # for cnt in contours:
        # #print(cv2.matchShapes(cnt, cross_contour[0], 1, 0.0))
        # pass

    # # 霍夫变换
    # lines = cv2.HoughLinesP(img_with_mask, 0.8, np.pi / 180, 90,
                        # minLineLength=50, maxLineGap=10)
    # drawing = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    # for line in lines:
        # x1, y1, x2, y2 = line[0]
        # cv2.line(drawing, (x1, y1), (x2, y2), (0, 0, 255), 2, lineType=cv2.LINE_AA)
    # cv2.imwrite('HoughLinesP.png', drawing)
    # 
    # # Shi-Tomasi角点检测
    # corners = cv2.goodFeaturesToTrack(img_with_mask, 100, 0.01, 10)
    # corners = np.int0(corners)
    # drawing = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    # for i in corners:
        # x, y = i.ravel()
        # cv2.circle(drawing, (x, y), 3, 255, -1)
    # cv2.imwrite('Tomasi.png', drawing)