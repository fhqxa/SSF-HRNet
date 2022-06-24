# SSF-HRNet
SSF-HRNet full name is Self-similarity feature based few-shot learning via hierarchical relation network.

# Datasets
Please click the Google Drive [link](https://drive.google.com/drive/folders/1sXJgi9pXo8i3Jj1nk08Sxo6x7dAQjf9u?usp=sharing) or [Baidu Drive (uk3o)](https://pan.baidu.com/s/17hbnrRhM1acpcjR41P3J0A) for downloading the 
following datasets, or running the downloading bash scripts in folder `datasets/` to download.

### Omniglot

### tieredImageNet


## Few-shot classification results
Experimental results on few-shot classification datasets.

<table>
  <tr>
    <td>datasets</td>
    <td colspan="4" align="center">Ominiglot</td>
  </tr>
  <tr>
    <td>setups</td>
    <td>5-way 1-shot</td>
    <td>5-way 5-shot</td>
  </tr>
  <tr>
    <td>Matching Network</td>
    <td align="center">98.10</td>
    <td align="center">98.90</td> 
  </tr>
  <tr>
    <td>Relation Network</td>
    <td align="center">99.60</td>
    <td align="center">99.80</td> 
  </tr>
  <tr>
    <td>SSF-HRNet</td>
    <td align="center">99.02</td>
    <td align="center">99.56</td> 
  </tr>
</table>

<table>
  <tr>
    <td>datasets</td>
    <td colspan="2" align="center">tieredImageNet</td>
  </tr>
  <tr>
    <td>setups</td>
    <td>5-way 1-shot</td>
    <td>5-way 5-shot</td>
  </tr>
  <tr>
    <td>Relation Network</td>
    <td align="center">54.26</td>  
    <td align="center">71.34</td>
  </tr>
    <tr>
    <td>HMRN</td>
    <td align="center">57.98</td>  
    <td align="center">74.70</td>
  </tr>
   <tr>
    <td>SSF-HRNet</td>
    <td align="center">58.68</td>
    <td align="center">74.47</td>
  </tr>
</table>


## Acknowledgment
Our project references the codes in the following repos.
- [Relation Network](https://github.com/floodsung/LearningToCompare_FSL)
- [HMRN](https://github.com/fhqxa/HMRN)
- [RENet](https://github.com/dahyun-kang/renet)
