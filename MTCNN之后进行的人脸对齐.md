### 可以用opencv进行仿射变化，比较简单的实现就是通过两个眼部位置计算角度做旋转，简单的代码贴在下面。
#### 代码：
    `def warp_affine(image, points, scale=1.0):
        eye_center = ((points[0][0] + points[1][0]) / 2,(points[0][1] + points[1][1]) / 2)
        dy = points[1][1] - points[0][1]
        dx = points[1][0] - points[0][0]
        # 计算旋转角度
        angle = cv2.fastAtan2(dy, dx)
        rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)
        rot_img = cv2.warpAffine(image, rot, dsize=(image.shape[1], image.shape[0]))
        plt.imshow(rot_img)
        plt.show()
        return rot_img****`