Search.setIndex({docnames:["bigfileops","hypercat","imageops","index","instruments","interferometry","ioops","loggers","morphology","ndiminterpolation","plotting","psf","units","utils"],envversion:53,filenames:["bigfileops.rst","hypercat.rst","imageops.rst","index.rst","instruments.rst","interferometry.rst","ioops.rst","loggers.rst","morphology.rst","ndiminterpolation.rst","plotting.rst","psf.rst","units.rst","utils.rst"],objects:{"":{bigfileops:[0,0,0,"-"],hypercat:[1,0,0,"-"],imageops:[2,0,0,"-"],instruments:[4,0,0,"-"],interferometry:[5,0,0,"-"],ioops:[6,0,0,"-"],loggers:[7,0,0,"-"],morphology:[8,0,0,"-"],ndiminterpolation:[9,0,0,"-"],plotting:[10,0,0,"-"],psf:[11,0,0,"-"],units:[12,0,0,"-"],utils:[13,0,0,"-"]},"bigfileops.CheckListSelector":{__init__:[0,2,1,""],draw_canvas:[0,2,1,""],run:[0,2,1,""],update_footer:[0,2,1,""]},"hypercat.ModelCube":{__init__:[1,2,1,""],get_image:[1,2,1,""],parameter_values:[1,4,1,""],print_sampling:[1,2,1,""]},"hypercat.Source":{__call__:[1,2,1,""],__init__:[1,2,1,""]},"imageops.Image":{F:[2,4,1,""],__init__:[2,2,1,""],getBrightness:[2,2,1,""],getTotalFluxDensity:[2,2,1,""],setBrightness:[2,2,1,""]},"imageops.ImageFrame":{I:[2,4,1,""],__init__:[2,2,1,""],changeFOV:[2,2,1,""],resample:[2,2,1,""],rotate:[2,2,1,""],setFOV:[2,2,1,""],setPixelscale:[2,2,1,""]},"instruments.Instrument":{__init__:[4,2,1,""],observe:[4,2,1,""]},"instruments.Interferometer":{__call__:[4,2,1,""],__init__:[4,2,1,""]},"instruments.Telescope":{__call__:[4,2,1,""],__init__:[4,2,1,""]},"ioops.FitsFile":{__init__:[6,2,1,""],get:[6,2,1,""],getdata:[6,2,1,""],getheader:[6,2,1,""],info:[6,2,1,""]},"loggers.LogFormatter":{__init__:[7,2,1,""],format:[7,2,1,""]},"morphology.Moment":{__call__:[8,2,1,""],__init__:[8,2,1,""]},"morphology.MomentAnalytics":{__call__:[8,2,1,""],__init__:[8,2,1,""],set_moment_name:[8,2,1,""]},"ndiminterpolation.NdimInterpolation":{__call__:[9,2,1,""],__init__:[9,2,1,""],get_coords:[9,2,1,""],serialize_vector:[9,2,1,""]},"psf.PSF":{__init__:[11,2,1,""],convolve:[11,2,1,""]},bigfileops:{CheckListSelector:[0,1,1,""],getIndexLists:[0,3,1,""],get_bytes_human:[0,3,1,""],get_bytesize:[0,3,1,""],get_hyperslab_via_mesh:[0,3,1,""],isragged:[0,3,1,""],loadjson:[0,3,1,""],memmap_hdf5_dataset:[0,3,1,""],storeCubeToHdf5:[0,3,1,""],storejson:[0,3,1,""]},hypercat:{ModelCube:[1,1,1,""],Source:[1,1,1,""],get_Rd:[1,3,1,""],get_clean_file_list:[1,3,1,""],get_pixelscale:[1,3,1,""],get_sed_from_fitsfile:[1,3,1,""],lum_dist_to_pixelscale:[1,3,1,""],mirror_all_fitsfiles:[1,3,1,""],mirror_fitsfile:[1,3,1,""]},imageops:{Image:[2,1,1,""],ImageFrame:[2,1,1,""],add_noise:[2,3,1,""],checkImage:[2,3,1,""],checkInt:[2,3,1,""],checkOdd:[2,3,1,""],computeIntCorrections:[2,3,1,""],make_binary:[2,3,1,""],measure_snr:[2,3,1,""],resampleImage:[2,3,1,""],resample_image:[2,3,1,""],rotateImage:[2,3,1,""],trim_square:[2,3,1,""]},instruments:{Instrument:[4,1,1,""],Interferometer:[4,1,1,""],Telescope:[4,1,1,""]},interferometry:{correlatedflux:[5,3,1,""],fft_pxscale:[5,3,1,""],ima2fft:[5,3,1,""],ima_ifft:[5,3,1,""],interferometry:[5,3,1,""],plot_inter:[5,3,1,""],uvload:[5,3,1,""]},ioops:{FitsFile:[6,1,1,""],save2fits:[6,3,1,""]},loggers:{LogFormatter:[7,1,1,""]},morphology:{Moment:[8,1,1,""],MomentAnalytics:[8,1,1,""],circle:[8,3,1,""],eq11:[8,3,1,""],findEmissionCentroid:[8,3,1,""],findEmissionPA:[8,3,1,""],findOrientation_loop:[8,3,1,""],fluxdensity_i_wave:[8,3,1,""],gaussian:[8,3,1,""],getAngle:[8,3,1,""],getImageEigenvectors:[8,3,1,""],getUnitVector:[8,3,1,""],get_all_moments_raw_matmul:[8,3,1,""],get_angle:[8,3,1,""],get_centroid:[8,3,1,""],get_cov_from_moments:[8,3,1,""],get_eigenvalues:[8,3,1,""],get_elongation:[8,3,1,""],get_hu_moment:[8,3,1,""],get_moment:[8,3,1,""],get_moment_central:[8,3,1,""],get_moment_central_matmul:[8,3,1,""],get_moment_old:[8,3,1,""],get_moment_raw:[8,3,1,""],get_moment_raw_matmul:[8,3,1,""],get_moment_scaleinvariant:[8,3,1,""],get_power:[8,3,1,""],get_wavelet:[8,3,1,""],get_wavelet_elliptical_mexh:[8,3,1,""],get_wavelet_elliptical_mexh_gaillot:[8,3,1,""],get_wavelet_elliptical_mexh_gaillot_full:[8,3,1,""],get_wavelet_elliptical_mexh_vuong:[8,3,1,""],get_wavelet_elliptical_mexh_vuong_fast:[8,3,1,""],gini:[8,3,1,""],imageToEigenvectors:[8,3,1,""],plot:[8,3,1,""],plot_symmetric1:[8,3,1,""],ratio_fluxdensity_upper_over_lower:[8,3,1,""],rectangle:[8,3,1,""],rot90ccw:[8,3,1,""],rotateVector:[8,3,1,""],square:[8,3,1,""],symmetric1:[8,3,1,""],test1:[8,3,1,""],whichside:[8,3,1,""],work:[8,3,1,""]},ndiminterpolation:{NdimInterpolation:[9,1,1,""]},plotting:{multiplot:[10,3,1,""],plotPanel:[10,3,1,""],plot_with_wcs:[10,3,1,""]},psf:{PSF:[11,1,1,""],getPSF:[11,3,1,""],get_normalization:[11,3,1,""],loadPSFfromFITS:[11,3,1,""],modelPSF:[11,3,1,""]},units:{UNITS:[12,5,1,""],getQuantity:[12,3,1,""],getValueUnit:[12,3,1,""],list_recognized_units:[12,3,1,""],numeric_pattern:[12,5,1,""]},utils:{arrayify:[13,3,1,""],get_wcs:[13,3,1,""],mirror_axis:[13,3,1,""],seq2str:[13,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:data"},terms:{"15391265e":8,"15m":11,"160n":1,"1e12":1,"1e45":1,"1st":10,"1x1":10,"1x3":10,"2008apj":1,"2008b":1,"24790402e":8,"2e45":13,"2nd":[8,10],"2x2":10,"3x1":10,"41661637e":8,"47109707e":8,"49580804e":8,"5um":11,"83323273e":8,"88438828e":8,"94219414e":8,"abstract":1,"boolean":10,"byte":0,"case":[0,4,6],"class":[0,1,2,4,6,7,8,9,10,11,12,13],"default":[0,1,2,4,5,6,8,9,10,13],"final":[2,4],"float":[0,1,2,5,8,10,11],"function":[1,2,6,8,9,10,11,13],"import":[0,6,10,13],"int":[0,1,2,6,8,9,11,13],"new":[0,2,6],"return":[0,1,2,4,5,6,8,9,10,11,12,13],"super":10,"true":[0,1,2,5,6,8,10,13],But:0,For:[0,6],RHS:8,The:[0,1,2,4,5,9,10,11,13],Then:[1,2],These:1,Uses:[1,8],WCS:[1,6,10,13],Will:0,With:13,__call__:[1,4,8,9],__init__:[0,1,2,4,6,7,8,9,11],_getimaget:2,abov:2,abs:[1,5],absolut:5,access:[0,1,8],accord:[0,2],accumul:2,accur:[2,8],accuraci:9,actual:[2,10],adapt:4,add:[2,10],add_nois:2,added:[0,9,10],adjust:2,adsab:1,affect:2,after:[0,2],against:12,agn:[1,9],airi:4,akin:2,albeit:10,alia:[1,2],all:[0,1,2,4,6,8,9,10,13],allow:1,along:[0,1,2,8,9],alreadi:[0,6],also:[0,1,2,4,6,11],alwai:[8,13],ambigu:2,analysi:8,angl:[1,2,5,8],angstrom:12,angular:[1,2,8,11,12],ani:[1,2,4,9,10,13],anoth:8,answer:8,anti:[1,2],antiparallel:8,apertur:4,apparatu:4,append:[0,6],appli:4,applic:[2,9],appropri:1,arai:8,arang:[0,13],arbitrari:9,arcsec2:[2,12],arcsec:[1,2,10,11,12],area:[1,2],arg:1,argument:[2,8],around:2,arr:[0,8,13],arr_i:[],arrai:[0,1,2,5,6,8,9,11,13],arrang:13,arraui:[],arrayifi:13,ascend:[8,9],ask:10,assert:2,assign:2,associ:0,assum:[0,2,6,10,11,13],astronom:[2,4],astrophys:1,astropi:[6,10,11,12,13],attach:4,automat:1,awai:2,axes:[1,8,9,10,13],axi:[0,1,2,8,9,10,13],axial:1,back:1,bak:1,base:[0,1,2,4,6,7,8,9,10,11],baselin:[4,5],becaus:2,befor:[0,1,4],behavior:13,being:[0,13],below:[0,1,2,6],best:1,better:9,between:[2,8],bicub:10,bigfileop:3,binari:[2,6],binariefi:2,binarifi:2,bit:[0,9],bolometr:1,bool:[0,1,2,5,6,8,10,13],both:[2,6,12],bottom:[2,10],box:0,bright:[2,4,8,12],brightness_unit:[1,2],bring:2,bytes:0,call:[1,4,6,8,10],camera:4,can:[0,1,2,4,6,8,9,10,11,12,13],capit:6,card:6,carri:10,categori:12,caus:[6,10],caution:2,ccw:8,cdot:[],center:2,central:[2,8,13],centroid:8,certain:0,chang:2,changefov:2,check:[0,2,4,12],checkbox:0,checkimag:2,checkint:2,checklistselector:0,checkodd:2,circinus_burtscher_2013:6,circl:8,clddata:1,clear:8,clock:8,clockwis:[1,2,8],close:2,clumpi:[1,5,9,11],cmap:10,coeffici:8,color:10,colorba:[],colorbar:10,colormap:10,column:[10,13],com:[],combin:9,come:1,comment:6,common:[2,4],complex:5,complic:9,comput:[0,1,2,5,8,9,11],computeintcorrect:2,configur:4,conform:1,consecut:2,conserv:2,consid:8,constant:[2,11],construct:[1,6,9,13],contain:[0,1,4,8],containgin:5,content:[0,3],contour:10,contstruct:0,conveni:[1,6],convers:5,convert:[0,1,2,9,10],convolut:[4,11],convolv:[4,11],convolved_imag:11,coord:9,coordin:[6,8,9,13],coords_pix:9,coorinates_pix:9,copi:[0,4],core:0,correct:[2,13],correl:[4,5],correlatedflux:5,correspond:[0,1,4,6,9,10,13],corrflux:5,could:12,count:[0,8],counter:[8,9],cours:9,cov:8,covari:8,creat:[0,1,6,11,12,13],crop:2,cube:[0,1,8,13],cubic:[2,9],cumpi:11,cunit:12,current:[0,1,2,9,10],currents:0,cut:2,data:[0,1,2,6,9,13],data_hypercub:9,datacub:9,dataset:0,deactiv:9,dec:[6,13],decep0:6,decreas:[2,4],definit:[2,13],deg:[1,2,8,12,13],degre:[2,8,9,12],delta:1,demonstr:10,densiti:[2,4],descend:8,describ:[10,13],descript:8,desir:[0,1,2,10,12],detail:[1,8],detector:4,determin:10,deviat:2,dialog:[0,1],diamet:[4,11],dict:[0,4,11],dictat:10,dictionari:[0,4,12],differ:[2,10],dim:[0,1,8,9],dimens:[2,8,9,13],dimension:[0,1,8,9],direct:[2,5,8,10,13],directori:1,discret:[2,8],dish:4,disk:[0,1],displai:[0,10],dist:1,distanc:[1,2,13],distribut:4,doc:13,docstr:[0,1,2,10],doe:[0,1,2,6],doesn:10,domin:8,don:1,done:1,dot:8,down:2,downsampl:2,draw_canva:0,dset:0,dsetpath:0,dsmemap:0,dst:0,due:[1,2],dure:0,dust:1,each:[0,8,9,10,13],easili:2,east:[1,2],edu:[1,4],effect:[2,4],eigentvector:8,eigenvalu:8,eigenvector:8,either:[1,2,4,6,9,10],elemen:0,element:[0,9,10,13],els:9,emb:2,emiss:8,empti:[2,6,10],enabl:4,end:[0,1,4],enforce2d:2,entir:[6,10,13],entri:[0,1,9],envelop:2,eps:[2,8],eq11:8,equival:12,erg:[1,12,13],error:2,especi:9,estim:5,etc:[0,2,4,6,12],eval:8,evalu:12,evec:8,everi:[0,1],exactli:2,exampl:[0,1,2,5,6,8,9,10,11,12,13],except:[0,2],exist:[0,6],expand:1,expect:13,exploit:8,expon:[0,12],ext:4,extend:0,extens:4,extent:10,extra:13,extra_keyword:6,extract:[1,4],eye:8,f_lambda:1,factor:2,fail:[0,2],fals:[0,2,5,6,8,10,13],faster:8,featur:[1,8],fetch:6,few:[2,10],fft:[4,5],fft_pxscale:5,fftscale:5,field:[2,6],figtitl:10,figur:10,file:[0,1,4,5,6,11,13],filenam:5,fill:[2,10,13],find:[2,8],findemissioncentroid:8,findemissionpa:8,findorientation_loop:8,first:[6,8,9,10],fit:[1,4,6,13],fitsfil:[1,6,11],five:[],flag:8,flexibl:[1,2,10],flip:1,float32:0,flux:[2,4,5],fluxdens:[2,12],fluxdensity_i_wav:8,fmt:1,folder:5,follow:12,format:[0,7,9,12,13],formatt:7,formula:8,fov:[2,4],fraction:8,frame:2,frequenc:5,from:[0,1,2,4,5,6,8,9,10,11,13],full:[0,1,9],fulli:1,func:[],gaussian:[2,4,8],gener:[4,9],geometr:8,geometri:10,get:[1,2,6,8],get_all_moments_raw_matmul:8,get_angl:8,get_bytes:0,get_bytes_human:0,get_centroid:8,get_clean_file_list:1,get_coord:9,get_cov_from_mo:8,get_eigenvalu:8,get_elong:8,get_hu_mo:8,get_hyperslab_via_mesh:0,get_imag:[1,2,11],get_moment:8,get_moment_centr:8,get_moment_central_matmul:8,get_moment_old:8,get_moment_raw:8,get_moment_raw_matmul:8,get_moment_scaleinvari:8,get_norm:11,get_pixelscal:1,get_pow:8,get_rd:1,get_sed_from_fitsfil:1,get_wavelet:8,get_wavelet_elliptical_mexh:8,get_wavelet_elliptical_mexh_gaillot:8,get_wavelet_elliptical_mexh_gaillot_ful:8,get_wavelet_elliptical_mexh_vuong:8,get_wavelet_elliptical_mexh_vuong_fast:8,get_wc:13,getangl:8,getbright:2,getdata:6,gethead:6,getimageeigenvector:8,getindexlist:0,getpsf:11,getquant:12,gettotafluxdens:2,gettotalfluxdens:[2,4],getunitvector:8,getvalueunit:[2,12],gini:8,give:[2,13],given:[0,4,8,9,10,11],gmail:[],gnomon:13,gpc:12,gracefulli:0,greater:9,grid:9,group:0,groupnam:0,grow:9,h5py:0,half:1,handl:[0,1,2],hardi:4,harvard:1,has:[6,9,10,11],have:[0,1,4,6,10,13],hdf5:[0,1],hdf5file:0,hdffile:[0,1],hdu:[1,4,6],hdukw:4,hdulist:6,header:[6,13],help:[2,8],here:[1,10,13],high:2,hold:[1,9],html:4,http:[1,4,13],human:0,hyper:1,hypercat:[0,2,6,11,13],hypercat_20170109:[0,1],hypercat_20170714:1,hypercub:[0,1,9,13],hypercubenam:0,hypercubeshap:0,hyperslic:1,i_i:[],i_j:1,identifi:12,idx:[0,6],idxlist:0,ifiabl:13,ight:[],ima2fft:5,ima:5,ima_fft:5,ima_ifft:5,imag:[1,2,4,5,6,8,9,10,11,13],imagefram:[2,10,11],imagehdu:6,imageop:[3,6,10,11,13],imagetoeigenvector:8,imes:1,img:[2,8,10],imgdata:[0,1],implement:1,implicit:1,implicitli:11,improv:9,imshow:10,inch:10,includ:10,incomplet:2,increas:[2,4],inde:0,index:[0,1,3,6,8,9,13],indic:[0,1,8],inexact:2,info:6,inher:1,inherit:13,init:6,initi:[2,8,9],inits:0,input:[2,4,13],instanc:[0,1,2,4,6,8,9,10,11,12,13],instanti:[1,4],instead:2,instrument:[3,6,11],integ:[2,8],integr:[1,2],interact:[0,1],interferomet:4,interferometri:3,interpol:[1,2,9,10],inverselinear:12,iofit:5,ioop:[3,13],isinst:9,israg:0,item:[],itertool:9,its:[1,2,11,12],itself:9,john:4,join:[2,13],json:[0,1],jsonfil:0,jstr:13,just:[8,13],keep:[8,9],kei:[4,6],keyword:[4,6],kind:4,kpc:12,kwarg:[4,10],label:10,lambda:[1,2,11],larg:11,larger:[0,2,10],largest:[0,2,8],last:10,later:1,launch:1,layout:10,leav:10,left:[1,2,8,10],leftmost:10,len:[9,10,13],lenght:9,length:[0,1,4,8,9,12],level:0,lies:8,like:6,lin:10,linarli:[],line:10,linear:[1,2,9,12],linearli:10,linsiz:1,list:[0,1,2,6,9,10,13],list_recognized_unit:12,listedcolormap:10,load:[0,1,4,5,11,13],loadfil:8,loadjson:0,loadpsffromfit:11,locat:[0,9],log10:9,log:[0,7,9,10],logarithm:10,logformatt:7,logger:3,logic:10,lol:0,loop:9,lsun:[1,12],lum:1,lum_dist_to_pixelscal:1,luminos:[1,12,13],lyr:12,m00:8,made:[0,4],magnitud:9,mai:2,make:[0,2],make_binari:2,mani:[2,9],mantissa:0,map:[0,5,9],margin:2,mas:[2,4,10,12],mass:8,match:6,math:[],mathemat:8,matplotlib:10,matrix:[8,9],max:[2,8,10],maxim:2,mean:[2,9,10],meant:1,measur:[2,8],measure_snr:2,mem:0,member:[0,6,10,13],memmap:0,memmap_hdf5_dataset:0,memori:0,mesh:0,meter:11,method:8,micron:[1,11,12],middl:10,might:2,milliarcsecond:12,minim:6,mirror:[1,13],mirror_all_fitsfil:1,mirror_axi:13,mirror_fitsfil:1,mjy:[2,10,12,13],mmat:[],mode:[0,6,9],model:[1,2,4,5,8,9,11],modelcub:[0,1],modelpsf:11,modul:3,moment:8,momentanalyt:8,more:[8,9,10],morpholog:3,mount:4,mpc:[1,12,13],mperrin:4,mpq:8,mu00:8,mu02:8,mu20:8,multi:[1,2,9,10],multipanel:10,multipl:[2,8],multiplot:10,mupq:8,must:[2,4,9,10,13],myfitsfil:6,name:[0,1,4,5,6],natur:10,naxi:6,nbyte:0,ncol:10,ndim:8,ndiminterpol:3,ndinterpol:1,nearest:2,necessari:2,necessarili:[2,9],need:[9,10,12],neg:[0,1,8],negat:2,nenkova:[1,9],nest:1,newcub:13,newfactor:2,newimag:2,newli:0,newnpix:2,newpixelscal:2,next:0,ngc1068:13,nhypercub:0,nikutta:[],nlambda:2,nois:2,noise_pattern:2,noisy_imag:2,non:[2,8],none:[0,1,2,4,6,8,10,13],norm:8,normal:[10,11],north:[1,2,8],notat:8,note:[0,2,10],noth:[0,2,6,10],now:1,npix:[1,2,8,11,13],npix_new:2,npixin:2,npixout:2,npz:8,nrow:10,num:12,number:[0,1,2,4,6,8,10,12,13],numer:[0,4,12],numeric_pattern:12,numpi:0,nxm:8,object:[0,1,2,4,6,8,9,10,12],objectnam:[1,13],obs:[6,10],observ:[1,4],obtain:[9,11],odd:[2,13],oder:8,oifit:[5,6],oiftisfil:6,old:9,omit:[0,1],onc:[1,8],one:[1,2,6,9,10,12,13],onedarrai:9,ones:[2,10],onli:[1,5,6,9,10],open:[0,6],oper:9,optic:4,option:[1,4,6,12],order:[2,8,9,10],org:13,origin:[1,2,8],other:[2,6,8,9,10],otherwis:[0,2,6,8,10],output:[1,2,13],outsid:2,outunit:1,over:[1,2,9],overal:9,oxford:4,padarrai:0,page:3,panel:10,panels:10,parallel:8,paramet:[0,1,2,4,5,6,8,9,10,11,12,13],parameter_valu:1,paramnam:0,parnam:0,part:[2,12],pass:2,path:[0,1,4,6],pattern:[4,12],peak:8,peak_pixel_bright:2,per:[1,2,11,12],perform:[0,2,4,9],perman:2,permiss:10,permit:12,phi:5,physic:1,pix:[2,4,12],pixel:[1,2,4,5,8,9,11],pixelarea:2,pixels:2,pixelscal:[1,2,4,11,13],pixelscale_detector:4,pixelscalekw:4,pixelscl:4,place:1,plane:[5,6],pleas:1,plot:[3,8,13],plot_int:5,plot_symmetric1:8,plot_with_wc:[10,13],plotpanel:10,plt:10,pmax:8,point:[0,5],pointer:0,popul:10,posit:[0,1,5,8],possibl:[0,2,6,10,13],postion:8,postiv:8,power:0,pre:0,preced:2,prefix:[0,8],present:[0,6,10],preserv:[2,4],previou:2,primaryhdu:6,princip:13,print:[0,1,2,6,12,13],print_sampl:1,process:4,prod:[0,9,10,13],produc:11,product:[8,9],project:13,proper:[4,6],properli:9,properti:[0,1,2,6],provid:[0,1,2,12],psf:[3,4,6,10],psf_conv:11,psf_miri_f1000w:4,psf_model:11,psfdict:[4,11],pylab:10,python:8,qualifi:[0,6],qualiti:2,quantiti:[2,10,11,12],question:8,quit:[2,9],r_0:11,r_o:11,rac:[],rad:12,radian:12,radiu:[1,8],raep0:6,rag:0,rais:[0,2],ram:[0,1],rang:11,rather:2,ratio:[4,11],ratio_fluxdensity_upper_over_low:8,raw:[2,8,11],raw_imag:2,read:[0,5,6],readabl:0,real:9,recogn:[2,12],recognized_unit:12,recommend:9,recomput:2,record:[0,6,7],rectangl:8,rectilinear:9,reduc:[0,1],refer:4,regex:12,regular:9,rel:8,relat:8,relev:0,remain:2,remov:[0,2],renorm:2,repeat:1,repres:[0,8,12],represent:12,request:[1,2],requir:2,res:2,resampl:[2,4],resample_imag:2,resampleimag:2,resamplingfactor:2,reshap:[0,9,13],resiz:0,resolv:[1,13],respect:1,result:[1,2,4,8,9],returnimag:2,returns:2,rework:9,right:[1,2,8,10],robert:[],root:0,rot90ccw:8,rotat:[1,2,8,13],rotateimag:2,rotatevector:8,rotimag:2,round:2,row:10,run:[0,1],rvec:8,s__author__:[],same:[0,1,2,10],sampl:[2,9],save2fit:[6,13],save:[6,13],save_backup:1,savefil:8,scale:[2,4,5,10,11],scaleinvari:8,scienc:4,screen:0,search:3,section:13,sed:1,see:[0,1,2,6,8,10,13],seem:1,sel:0,selarr:0,selec:0,select:[0,1],self:[0,1,2,4,8,9,13],sens:8,separ:[0,6,10,12],seq2str:13,seq:[0,1,6,10,12,13],sequenc:[0,1,6,10,12,13],sequenti:6,seri:4,serialize_vector:9,session:1,set:[2,9,13],set_moment_nam:8,setbright:2,setfov:2,setpixelscal:2,sever:[4,6,9],shape:[0,1,2,8,9,10,13],shape_:9,sharei:10,sharex:10,should:[0,1,2],show:10,sig:[0,1,2,8,9],sigma:8,sigmai:8,sigmax:8,sign:12,signal:2,signatur:[2,8],significand:0,silic:1,simpl:[1,12],simpli:[0,1],simul:4,sinc:[2,10],singl:[0,4,9,10,13],size:[0,1,2,9,10,11],skip:1,sky:[1,4,5,6,10,13],slab:1,slice:1,slightli:2,slower:9,small:2,smaller:[2,10],snr:[1,2,4],softwar:4,solid:2,sollum:12,some:[2,6],someth:0,sort:[1,8],sortdescend:8,sourc:[0,1,2,4,5,6,7,8,9,10,11,12,13],space:[5,9,10],span:9,specif:11,specifi:[1,2,12],spline:[2,9],split:12,squar:[1,2,8],stabl:13,standard:[1,2],statu:10,std:2,stdout:6,step:0,still:2,storag:1,store:[0,1],storecubetohdf5:0,storejson:0,str:[0,1,2,4,5,6,9,10,11,12,13],strehl:[4,11],strictli:2,string:[0,1,4,6,9,10,12,13],stsci:4,sub:[0,1,13],subcub:1,subcube_select:1,subcube_selection_sav:1,subgroup:0,sublim:1,subsequ:6,successfulli:13,suffix:[0,1],suitabl:[11,13],sum:[2,8],sum_i:[],sum_j:1,summat:8,symmertric1:8,symmetri:1,symmetric1:8,system:[6,13],tabl:[6,12],take:[1,2,9],tan:13,target:2,tauv:[1,2],tbd:11,telescop:[4,6,11],temperatur:[1,12],ten:10,termin:1,test1:8,test:[0,2,9],than:[0,2,8,9,10],thei:[4,10],them:[1,10],theta:[0,1,2,8,9],theta_i:9,thi:[0,1,2,4,6,8,9,13],third:2,those:10,thresh:8,thu:2,tick:10,time:[0,1,8],titl:10,tmt:4,tmt_iri:4,tmt_michi:4,tool:4,top:[2,10],tori:9,toru:5,total:[0,2,4],total_flux_dens:[1,2,13],totalfluxdens:2,touch:[1,2],toward:2,transpos:13,trim:2,trim_squar:2,truncat:[2,10,13],tsub:1,tupl:[0,1,6,8,9,10,12,13],turn:[0,13],two:[1,8,10,13],type:[0,1,2,4,5,6,8,9,10,11,12,13],typic:4,u_px:5,uji:[2,12],unchang:[0,2],under:[0,1],unequ:0,uniqu:9,unit:[0,1,2,3,4,8,10,11],unselect:[0,1],updat:1,update_foot:0,upper:10,upsampl:2,use:[0,1,2,9,10],used:[0,2,4,9,10,13],user:[0,2],uses:2,usewc:6,using:[0,2,9,11],usual:13,util:3,uvec:8,uvfilenam:5,uvload:5,uvplan:5,v_px:5,val:8,valid:12,valu:[0,1,2,4,5,6,8,9,10,11,12,13],valueerror:[2,12],variant:1,variou:2,various:1,vcoord:6,vec:[8,9,13],vec_i:9,vector:[1,8,9,13],vectup:9,verbos:[6,8],veri:1,verifi:2,version:2,vertic:0,via:[0,1,8,9,13],view:[1,2],viridi:10,visibl:4,visir:4,vizier:[1,13],vor:8,wai:[8,9],warn:1,wave:[0,1,9,11,12],wavelength:[1,2,11],wcs:[6,10,13],webbpsf:4,west:[1,2],what:6,whatev:0,when:8,where:[0,1,6,8,9,12],whether:[2,10],which:[0,1,2,4,6,8,9,10,11],whichsid:8,whitespac:12,wise:8,within:[1,2],without:6,wordsiz:0,work:[0,8],world:[6,13],would:12,written:[0,1,6,9],www:4,xbar:8,xoff:8,ybar:8,yet:[0,1,2,4,6],yield:2,yoff:8,you:[4,6,10],your:5,zero:2},titles:["bigfileops module","hypercat module","imageops module","Welcome to HyperCAT API\u2019s documentation!","instruments module","interferometry module","ioops module","loggers module","morphology module","ndiminterpolation module","plotting module","psf module","units module","utils module"],titleterms:{api:3,bigfileop:0,document:3,hypercat:[1,3],imageop:2,indic:3,instrument:4,interferometri:5,ioop:6,logger:7,modul:[0,1,2,4,5,6,7,8,9,10,11,12,13],morpholog:8,ndiminterpol:9,plot:10,psf:11,tabl:3,unit:12,util:13,welcom:3}})