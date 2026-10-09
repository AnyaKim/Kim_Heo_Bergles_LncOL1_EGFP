//brains from C:\Users\User\.brainglobe\allen_mouse_25um_v1.2
//iterated this for slide 152, 175,197,326
run("Duplicate...", " ");
run("Duplicate...", " ");
run("Duplicate...", " ");
run("Duplicate...", " ");
run("Duplicate...", " ");

//477- striatal boundaries- yellow
selectImage("MAX_annotation.tiff");
changeValues(477, 477, 614454280);
setThreshold(614454272, 614454272, "raw");
run("Convert to Mask");
run("16-bit");
changeValues(255, 255,40);

//275- Lateral septal complex-green
selectImage("MAX_annotation-1.tiff");
list = newArray
(275,242,310,333,250,258,266);
for ( i=0; i<list.length; i++ ) { 
	changeValues(list[i], list[i], 614454280);
	}
setThreshold(614454272, 614454272, "raw");
run("Convert to Mask");
run("16-bit");
changeValues(255, 255,40);

//278- Striatum like amygdala complex- not in any of the selected slices
//selectImage("MAX_annotation-2.tiff");
//list = newArray
//(23,292,403,536,1105,544,551,559,278);
//for ( i=0; i<list.length; i++ ) { 
//	changeValues(list[i], list[i], 614454280);
//	}
//setThreshold(614454272, 614454272, "raw");
//run("Convert to Mask");
//run("16-bit");
//changeValues(255, 255,40);

//substantia nigra- red- NOT striatum- just highlighting green projection area
selectImage("MAX_annotation-2.tiff");
changeValues(381, 381, 614454280);
setThreshold(614454272, 614454272, "raw");
run("Convert to Mask");
run("16-bit");
changeValues(255, 255,40);

//485- Striatum dorsal region-cyan
selectImage("MAX_annotation-3.tiff");
changeValues(485, 485, 614454280);
setThreshold(614454272, 614454272, "raw");
run("Convert to Mask");
run("16-bit");
changeValues(255, 255,40);

//493- Striatumventral region blue
selectImage("MAX_annotation-4.tiff");
list = newArray
(56,754,998,493);
for ( i=0; i<list.length; i++ ) { 
	changeValues(list[i], list[i], 614454280);
	}
setThreshold(614454272, 614454272, "raw");
run("Convert to Mask");
run("16-bit");
changeValues(255, 255,40);
	
//672- Caudoputamen- magenta
selectImage("MAX_annotation-5.tiff");
changeValues(672, 672, 614454280);
setThreshold(614454272, 614454272, "raw");
run("Convert to Mask");
run("16-bit");
changeValues(255, 255,40);

run("Merge Channels...", "c1=MAX_annotation-2.tiff c2=MAX_annotation-1.tiff c3=MAX_annotation-4.tiff c4=MAX_reference.tiff c5=MAX_annotation-3.tiff c6=MAX_annotation-5.tiff c7=MAX_annotation.tiff create");