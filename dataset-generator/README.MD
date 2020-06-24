Dataset generator for the optc box scanner.
It is able to generate x amount of image datasets with 25 randomly placed optc units.

I think the image manipulation portion of the code is pretty straightforward, so I won't leave much writing about it.
But I think annotation generation portion is a bit all over the place, so I will leave note here. I may completely forget what is going on later on, making this whole thing very hard to edit later on, if need be. 

This app's background is static as of now. This means THIS PROGRAM DOES NOT FUNCTION PROPERLY UNLESS USING THAT "baseBG.png" FILE IN THIS REPO. 
Entire program is built around it. In order to create dataset from different background image, following will need to change:
- unit placement locations in main.js 
- unit distance from one another in main.js 
- sample coco file that lists the bounding box for all units. See below regarding that. 


Annotation is generated in COCO format. 
In annotations folder, ybat-master is a tool I used in order generate the segmentation values for the unit locations. I have no idea how segementation and area values in COCO are created, so I generated them manually with this tool. 
If I want to change the location of the units' annotation, at least the segementation and area values will need to be retrieved from similar tools, which is why I bothered to leave it in the repo. 
If YOLO does not require those two values for our use case (areas we define as "unit" are just boxes anyway...) this step can be forgone altogether and that tool can be removed from this repo. I mean, I'd still need to grab bounding box values from somewhere, but that is a value that can probably derived mathmatically. 


With the ybat tool, what I do is first open it with ybat.html. 
Import sampleclasses.txt, which simply has numbers 1-25.
This allows me to assign the numbers to each annotation I draw. 
If new assignments need to be made, BE SURE TO GO FROM TOP TO BOTTOM, LEFT TO RIGHT:

i.e.:
1, 2, 3, 4, 5
6, 7, 8, 9, 10
so on and so forth. 

once all boxes are selected, export to COCO.
This will download a document called coco.json. Move this folder into the OPTC-BOX-DATASET-GENERATOR/annotations.
At this point running the main.js should supply with new datasets. 




number of datasets that this app generates can be altered on main.js. Find "dataSetAmount".
