class BBox:
    confidence = 1.0

    def __init__(self, classId, x1, y1, x2, y2, w, h, confidence=1.0):
        self.classId = int(classId)
        self.w = w
        self.h = h
        self.box = (x1, y1, x2, y2)
        self.confidence = confidence

    def getYolo(self):
        x1, y1, x2, y2 = self.box
        x = (x1 + x2) / (2.0 * (self.w or 1))
        y = (y1 + y2) / (2.0 * (self.h or 1))
        w = (x2 - x1) / float(self.w)
        h = (y2 - y1) / float(self.h)

        return f"{self.classId} {x} {y} {w} {h}"

    def __repr__(self):
        return f"class={self.classId} box={self.box}"

    def show(self, frame, truth=True):
        cls = self.classId  # int(box.cls[0])
        if truth:
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
            if cls == 81:
                color = (0, 0, 255)

        confidence = round(self.confidence * 100)
        if truth:
            label = f"{cls}"
        else:
            label = f"{cls} {confidence:02}%"
        fsize, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        fw, fh = fsize
        delta = 2

        x1, y1, x2, y2 = map(int, self.box)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv.LINE_AA)

        top = y1
        if truth:
            top = y2 + fh + 2 * delta

        cv.rectangle(frame, (x1, top - fh - 2 * delta), (x2, top), color, -1)

        cv.putText(
            frame,
            label,
            (x1, top - delta),
            0,
            0.9,
            (0, 0, 0),
            2,
            lineType=cv.LINE_AA,
        )


class BBoxes:
    def __init__(self, truth=True):
        self.bboxes = []  # bboxes
        self.id = time.time()
        self.truth = truth

    def __repr__(self):
        result = [f"bboxes={self.id}"]
        for box in self.bboxes:
            result.append(str(box))
        return "\n".join(result)

    def add(self, bbox: BBox):
        self.bboxes.append(bbox)

    def extend(self, bboxes):
        for box in bboxes:
            self.add(box)

    def has(self, classId):
        if not isinstance(classId, list):
            classId = [classId]
        for bbox in self.bboxes:
            if bbox.classId in classId:
                return True
        return False

    def hasOnly(self, classId):
        if not isinstance(classId, list):
            classId = [classId]
        for bbox in self.bboxes:
            # print(classId, bbox.classId, classId!=bbox.classId)
            if not bbox.classId in classId:
                return False
        return True

    def import_txt(self, txtfile, w, h, classes={}):
        if not os.path.exists(txtfile):
            return []
        data = open(txtfile).readlines()
        for obj in data:
            classId, xc, yc, wf, hf = map(float, obj.strip().split(" "))
            wp = wf * w
            hp = hf * h
            x1 = xc * w - wp / 2
            y1 = yc * h - hp / 2
            self.bboxes.append(BBox(classId, x1, y1, x1 + wp, y1 + hp, w, h))
        return self.bboxes

    def show(self, frame):
        for box in self.bboxes:
            box.show(frame, self.truth)

    def merge(self, bboxes_in, filter_classes):
        """
        add to self.bboxes bboxes_in if bbox is in filter_classes replacing those in self

        """
        bboxes_result = BBoxes()

        # non abbiamo nulla da aggiungere
        if not filter_classes:
            return bboxes_in

        # rimuoviamo tutti i bbox che sono in filter_classes perchè quelli in bboxes_in sono prioritari
        for bbox in self.bboxes:
            if not bbox.classId in filter_classes:
                bboxes_result.add(bbox)

        # aggiungiamo quelli in bboxes_in se appartengono a filter_class
        for bbox_in in bboxes_in.bboxes:
            if bbox_in.classId in filter_classes:
                bboxes_result.add(bbox_in)

        return bboxes_result

    def save(self, frame, filename, path, include=[]):
        if not os.path.exists(path):
            os.makedirs(path)

        basename = os.path.basename(filename).split(".")[0]

        classes_txt = os.path.join(path, "classes.txt")
        if not os.path.exists(classes_txt):
            with open(classes_txt, "w") as f:
                f.write("\n".join([str(c) for c in include]))

        filename_new = os.path.join(path, basename + ".txt")
        filename_jpg = os.path.join(path, basename + ".jpg")
        with open(filename_new, "w") as f:
            log.info("save to %s", filename_new)
            for bbox in self.bboxes:
                if not include or bbox.classId in include:
                    f.write(bbox.getYolo() + "\n")
        cv.imwrite(filename_jpg, frame, [cv.IMWRITE_JPEG_QUALITY, 100])
