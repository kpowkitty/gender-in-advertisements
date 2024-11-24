# NSF LCCF Advanced Computing for Social Change 2024

## Fighting gender bias in advertisements
As a woman in computing, I am extremely passionate about getting more women in computing, especially with how little there seems to be.
There is 4 women to every 40 men in my comp sci classes, and even less when I was in high school -- I was the only one.
Growing up in the 2000's, I was told that computers were a "boy's toy". I never knew a girl who used a computer. It was extremely socially 
paralyzing, and if we are promoting them as a "boy's toy," we will see this reflected in our newer generations. 

## Gender in Advertisements
During my week at ACSC, I found a database by the NCES: computer science degrees conferred
to men and women across the United States since 1965. After visualizing this in Jupyter Notebooks
with TACC's Frontera super computer, I found two distinct moments:
 - 1980's<br>
     Both men and women CS degrees conferred took a nosedive
 - 2010's<br>
     Men recovered and grew exponentially, whereas women only just barely started to grow again
     in the 2020's

Like the scientist I am, I naturally asked the question...

## Why?
There are, of course, many reasons why. But, I wanted to zoom in on advertisements. Advertisements
  are extremely influential; razor companies promoted shaving as a feminine product to sell more razors...
  and it worked. So why couldn't computer advertisements be just as influential?

## How?
I hand collected data on computer advertisements, and found that gender was representation was unbalanced:
  In the 1980's, it was men's 40% to women's 10% prescence. If both a man and woman was present, it was about 
  30% -- but a man was required to be present. That means that men, in total, were present in 70% of ads compared
  to women's 40%. Since the 2010's, though, it has been much more balanced: 40% women to 30% men, with both present 
  being 6% -- almost completely split down the middle.

## The problem
Hand collecting data has many problems.
1. I can only look at so many advertisements, so it is hard to get accurate data. The few advertisements I look at
   could all include women, men, or none, and be inaccurate representations of the time.
3. There's no scalability.
4. There's no reusability.

## The solution: AI
I followed [this medium article](url) on how to train a Pytorch model to identify whether the image contains a male or a female.
During the conference, I only got it up and running with no new testing, so it was trained on 10 epochs with 32x32 resolution pictures. 
On the given database, it was 64% accurate; on my advertisements, it was even lower, at 41%.

## What's next?
I will be training it on higher image quality, with more epochs, on my AMD 6970 GPU. Already, I was able to train it to 75% accuracy
with 128x128 resolution on 10 epochs on my macbook air M3, but it has its limitations. So, I believe with my GPU, I can get it close 
to 100% accurate.

When it is fully trained, it will be able to detect whether a man or woman is present in a picture. With further adjustments, I would
like to be able to train it on other metrics: if both are present and if none are present. Then, it will match what I hand collected.

Furthermore, I would like it to be able to test if the pictures are more "female" oriented or "male" oriented by analyzing other
aspects of the pictures.

With this tool, we can ensure that there is no gender bias in advertisements again. This way, we can prevent similar effects on 
our younger generation from ever happening again!

## Guide
<b>Dependencies:</b> torch, torchvision, matplotlib, opendatasets, Jupyter Labs
1. First, clone the repository however you choose:<br>
    <code>git clone <ssh/https></code>
2. Open the project in Jupyter:<br>
    If you have Jupyter Labs:<br>
     <code>jupyter-lab</code><br>
    Or, if you have jupyter notebooks,<br>
     <code>jupyter-notebook</code>
3. Open gender_id.ipynb<br>

<b>To train your own model:</b><br>
- Download the training dataset:<br>
    <code>python dataset_dl.py</code>
- Change <code>t</code> to the size you desire.<br>
- Change class_finder to have the correct matrix multiplication according to what dimensions you are resizing it to be.<br>
- Change any other variables in gender_id.py that are expecting that resolution to match.<br>
- In gender_id.ipynb, run each cell one by one<br><br>

<b>To use the pre-trained 32x32 model:</b><br>
- In <code>gender_id.ipynb</code>, create new cells using the commented-out code at the bottom of <code>gender_id.py</code><br>
     (make sure you also include the imported libraries at the top of this file)<br>
- Redefine the path to your dataset using "DATA_DIR"<br>
     This model is expecting a batch_size of 32.<br>
- Run all cells

Once I get this running on my home computer, I will release the 128x128 model, and more!
