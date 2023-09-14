
import json
import os

import configargparse
import imagesize



# {
#   "id": 8212,
#   "data": {
#     "image": "/data/upload/5/7b4f48f3-0a0a00b2fbe89a47.jpg"
#   },
#   "annotations": [],
#   "predictions": []
# }


def initParser():
    parser = configargparse.get_argument_parser()
    parser.add_argument("txts", nargs="+", help="list of txt")
    parser.add_argument("--js", default="/tmp/project.json")
    parser.add_argument("--path", default="plates-01")

#    parser.add_argument("--save", action="store_true", default=False, help="save yolo dataset")
    return parser.parse_args()


def getResult(w=0,h=0,annotation={},classes=[]): 
    return {
                        "original_width": w,
                        "original_height": h,
                        "image_rotation": 0,
                        "value": {
                            "points": [(point[0]*100, point[1]*100) for point in annotation["points"]],
                            # [
                            #     [
                            #         29.766596974525473,
                            #         67.82036160061968
                            #     ],
                            #     [
                            #         29.388024224958762,
                            #         64.91666881082932
                            #     ],
                            #     [
                            #         43.46734227909765,
                            #         64.63458728479942
                            #     ],
                            #     [
                            #         43.63063990833586,
                            #         67.84785202298363
                            #     ]
                            # ],
                            "closed": True,
                            "polygonlabels": classes
                        },
                        #"id": "0_qsFlS4d7",
                        "from_name": "label",
                        "to_name": "image",
                        "type": "polygonlabels",
                        "origin": "manual"
         
                #"was_cancelled": False,
                #"ground_truth": True,
                # "created_at": "2023-09-11T21:01:47.413610Z",
                # "updated_at": "2023-09-12T11:58:12.907732Z",
                # "draft_created_at": "2023-09-11T21:01:37.526039Z",
                # "lead_time": 55.964,
                # "prediction": {},
                # "result_count": 0,
                # "unique_id": "cf485396-dda1-4197-a748-7775a72b4578",
                # "last_action": None,
                # "task": 8114,
                # "project": 4,
                # "updated_by": 3,
                # "parent_prediction": None,
                # "parent_annotation": None,
                # "last_created_by": None
            }

def getTemplate(id=0, w=0, h=0, classes=["plate"],
    imagefile="", path="plates-01", annotations=[]):
    
    
    results = [getResult(w=w, h=h, classes=classes, annotation=a) for a in annotations]
    

    
    return {
        #"id": id,
        "annotations": [{
            "result":results,
        }],
        #"file_upload": "e0ac581e-dima-panyukov-DwxlhTvC16Q-unsplash.jpg",
        "drafts": [],
        "predictions": [],
        "data": {
            "image": f"/data/local-files/?d=/{path}/{imagefile}"
        },
        # "meta": {},
        # "created_at": "2023-09-11T21:01:11.123773Z",
        # "updated_at": "2023-09-12T11:58:13.180389Z",
        # "inner_id": 1,
        "total_annotations": len(annotations),
        # "cancelled_annotations": 0,
        # "total_predictions": 0,
        # "comment_count": 0,
        # "unresolved_comment_count": 0,
        # "last_comment_updated_at": None,
        # "project": 4,
        # "updated_by": 3,
        # "comment_authors": []
    }

def readTxt(txt):
    results = []
    with open(txt) as f:
        for row in f.readlines():
            values = [pp.strip() for pp in row.split(" ")]
            result = {"class": values[0], "points":[]}
            
            i=1
            while i<len(values):
                result["points"].append(list(map(float, values[i:i+2])))
                i += 2
            results.append(result)
    return results

def main():
    config = initParser()

    dataset = []
    for id,txt in enumerate(config.txts):
        imagefile = os.path.splitext(txt)[0] + ".jpg"
        try:
            w, h = imagesize.get(imagefile)
        except:
            continue

        annotations = readTxt(txt)
        dataset.append(getTemplate(w=w, h=h, path=config.path, 
            imagefile=os.path.basename(imagefile),
            annotations=annotations
            ))

    with open(config.js, "w") as out:
        json.dump(dataset, out, indent = 2)
        print(f"saved to {config.js}")


if __name__ == "__main__":
    main()
