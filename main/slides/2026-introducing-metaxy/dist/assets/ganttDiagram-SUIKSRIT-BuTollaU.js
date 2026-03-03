import{g as ce,s as le,v as ue,t as de,a as fe,b as he,c as o,d as ut,e as Tt,L as me,M as ke,N as ye,f as ge,O as pe,P as ve,l as rt,Q as Te,R as Rt,S as Ht,T as be,U as xe,V as we,W as _e,X as De,Y as Se,Z as Me,$ as Bt,a0 as qt,a1 as Gt,a2 as Xt,a3 as Ut,a4 as Ce,k as Ee,A as Ie,n as at,a5 as Kt,r as Ye,u as Ae,a6 as Dt}from"./Mermaid.vue_vue_type_script_setup_true_lang-B3Ftbgu8.js";import"./modules/vue-aalH_SXY.js";import"./index-BMlti1zG.js";import"./modules/shiki-D_M5EFjr.js";import"./lz-string-BCxBxReP.js";import"./modules/file-saver-B7oFTzqn.js";var $e=Dt({"../../node_modules/.pnpm/dayjs@1.11.18/node_modules/dayjs/plugin/isoWeek.js"(t,r){(function(n,i){typeof t=="object"&&typeof r<"u"?r.exports=i():typeof define=="function"&&define.amd?define(i):(n=typeof globalThis<"u"?globalThis:n||self).dayjs_plugin_isoWeek=i()})(t,(function(){var n="day";return function(i,a,k){var y=o(function(F){return F.add(4-F.isoWeekday(),n)},"a"),_=a.prototype;_.isoWeekYear=function(){return y(this).year()},_.isoWeek=function(F){if(!this.$utils().u(F))return this.add(7*(F-this.isoWeek()),n);var x,L,V,N,z=y(this),S=(x=this.isoWeekYear(),L=this.$u,V=(L?k.utc:k)().year(x).startOf("year"),N=4-V.isoWeekday(),V.isoWeekday()>4&&(N+=7),V.add(N,n));return z.diff(S,"week")+1},_.isoWeekday=function(F){return this.$utils().u(F)?this.day()||7:this.day(this.day()%7?F:F-7)};var W=_.startOf;_.startOf=function(F,x){var L=this.$utils(),V=!!L.u(x)||x;return L.p(F)==="isoweek"?V?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):W.bind(this)(F,x)}}}))}}),Fe=Dt({"../../node_modules/.pnpm/dayjs@1.11.18/node_modules/dayjs/plugin/customParseFormat.js"(t,r){(function(n,i){typeof t=="object"&&typeof r<"u"?r.exports=i():typeof define=="function"&&define.amd?define(i):(n=typeof globalThis<"u"?globalThis:n||self).dayjs_plugin_customParseFormat=i()})(t,(function(){var n={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},i=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,a=/\d/,k=/\d\d/,y=/\d\d?/,_=/\d*[^-_:/,()\s\d]+/,W={},F=o(function(D){return(D=+D)+(D>68?1900:2e3)},"a"),x=o(function(D){return function(M){this[D]=+M}},"f"),L=[/[+-]\d\d:?(\d\d)?|Z/,function(D){(this.zone||(this.zone={})).offset=(function(M){if(!M||M==="Z")return 0;var O=M.match(/([+-]|\d\d)/g),A=60*O[1]+(+O[2]||0);return A===0?0:O[0]==="+"?-A:A})(D)}],V=o(function(D){var M=W[D];return M&&(M.indexOf?M:M.s.concat(M.f))},"u"),N=o(function(D,M){var O,A=W.meridiem;if(A){for(var j=1;j<=24;j+=1)if(D.indexOf(A(j,0,M))>-1){O=j>12;break}}else O=D===(M?"pm":"PM");return O},"d"),z={A:[_,function(D){this.afternoon=N(D,!1)}],a:[_,function(D){this.afternoon=N(D,!0)}],Q:[a,function(D){this.month=3*(D-1)+1}],S:[a,function(D){this.milliseconds=100*+D}],SS:[k,function(D){this.milliseconds=10*+D}],SSS:[/\d{3}/,function(D){this.milliseconds=+D}],s:[y,x("seconds")],ss:[y,x("seconds")],m:[y,x("minutes")],mm:[y,x("minutes")],H:[y,x("hours")],h:[y,x("hours")],HH:[y,x("hours")],hh:[y,x("hours")],D:[y,x("day")],DD:[k,x("day")],Do:[_,function(D){var M=W.ordinal,O=D.match(/\d+/);if(this.day=O[0],M)for(var A=1;A<=31;A+=1)M(A).replace(/\[|\]/g,"")===D&&(this.day=A)}],w:[y,x("week")],ww:[k,x("week")],M:[y,x("month")],MM:[k,x("month")],MMM:[_,function(D){var M=V("months"),O=(V("monthsShort")||M.map((function(A){return A.slice(0,3)}))).indexOf(D)+1;if(O<1)throw new Error;this.month=O%12||O}],MMMM:[_,function(D){var M=V("months").indexOf(D)+1;if(M<1)throw new Error;this.month=M%12||M}],Y:[/[+-]?\d+/,x("year")],YY:[k,function(D){this.year=F(D)}],YYYY:[/\d{4}/,x("year")],Z:L,ZZ:L};function S(D){var M,O;M=D,O=W&&W.formats;for(var A=(D=M.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,(function(v,p,g){var f=g&&g.toUpperCase();return p||O[g]||n[g]||O[f].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,(function(c,l,h){return l||h.slice(1)}))}))).match(i),j=A.length,B=0;B<j;B+=1){var E=A[B],T=z[E],d=T&&T[0],u=T&&T[1];A[B]=u?{regex:d,parser:u}:E.replace(/^\[|\]$/g,"")}return function(v){for(var p={},g=0,f=0;g<j;g+=1){var c=A[g];if(typeof c=="string")f+=c.length;else{var l=c.regex,h=c.parser,m=v.slice(f),b=l.exec(m)[0];h.call(p,b),v=v.replace(b,"")}}return(function(s){var P=s.afternoon;if(P!==void 0){var e=s.hours;P?e<12&&(s.hours+=12):e===12&&(s.hours=0),delete s.afternoon}})(p),p}}return o(S,"l"),function(D,M,O){O.p.customParseFormat=!0,D&&D.parseTwoDigitYear&&(F=D.parseTwoDigitYear);var A=M.prototype,j=A.parse;A.parse=function(B){var E=B.date,T=B.utc,d=B.args;this.$u=T;var u=d[1];if(typeof u=="string"){var v=d[2]===!0,p=d[3]===!0,g=v||p,f=d[2];p&&(f=d[2]),W=this.$locale(),!v&&f&&(W=O.Ls[f]),this.$d=(function(m,b,s,P){try{if(["x","X"].indexOf(b)>-1)return new Date((b==="X"?1e3:1)*m);var e=S(b)(m),w=e.year,$=e.month,Y=e.day,I=e.hours,q=e.minutes,C=e.seconds,Q=e.milliseconds,st=e.zone,ct=e.week,ht=new Date,mt=Y||(w||$?1:ht.getDate()),lt=w||ht.getFullYear(),R=0;w&&!$||(R=$>0?$-1:ht.getMonth());var U,G=I||0,nt=q||0,J=C||0,it=Q||0;return st?new Date(Date.UTC(lt,R,mt,G,nt,J,it+60*st.offset*1e3)):s?new Date(Date.UTC(lt,R,mt,G,nt,J,it)):(U=new Date(lt,R,mt,G,nt,J,it),ct&&(U=P(U).week(ct).toDate()),U)}catch{return new Date("")}})(E,u,T,O),this.init(),f&&f!==!0&&(this.$L=this.locale(f).$L),g&&E!=this.format(u)&&(this.$d=new Date("")),W={}}else if(u instanceof Array)for(var c=u.length,l=1;l<=c;l+=1){d[1]=u[l-1];var h=O.apply(this,d);if(h.isValid()){this.$d=h.$d,this.$L=h.$L,this.init();break}l===c&&(this.$d=new Date(""))}else j.call(this,B)}}}))}}),Le=Dt({"../../node_modules/.pnpm/dayjs@1.11.18/node_modules/dayjs/plugin/advancedFormat.js"(t,r){(function(n,i){typeof t=="object"&&typeof r<"u"?r.exports=i():typeof define=="function"&&define.amd?define(i):(n=typeof globalThis<"u"?globalThis:n||self).dayjs_plugin_advancedFormat=i()})(t,(function(){return function(n,i){var a=i.prototype,k=a.format;a.format=function(y){var _=this,W=this.$locale();if(!this.isValid())return k.bind(this)(y);var F=this.$utils(),x=(y||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,(function(L){switch(L){case"Q":return Math.ceil((_.$M+1)/3);case"Do":return W.ordinal(_.$D);case"gggg":return _.weekYear();case"GGGG":return _.isoWeekYear();case"wo":return W.ordinal(_.week(),"W");case"w":case"ww":return F.s(_.week(),L==="w"?1:2,"0");case"W":case"WW":return F.s(_.isoWeek(),L==="W"?1:2,"0");case"k":case"kk":return F.s(String(_.$H===0?24:_.$H),L==="k"?1:2,"0");case"X":return Math.floor(_.$d.getTime()/1e3);case"x":return _.$d.getTime();case"z":return"["+_.offsetName()+"]";case"zzz":return"["+_.offsetName("long")+"]";default:return L}}));return k.bind(this)(x)}}}))}}),We=Dt({"../../node_modules/.pnpm/dayjs@1.11.18/node_modules/dayjs/plugin/duration.js"(t,r){(function(n,i){typeof t=="object"&&typeof r<"u"?r.exports=i():typeof define=="function"&&define.amd?define(i):(n=typeof globalThis<"u"?globalThis:n||self).dayjs_plugin_duration=i()})(t,(function(){var n,i,a=1e3,k=6e4,y=36e5,_=864e5,W=/\[([^\]]+)]|Y{1,4}|M{1,4}|D{1,2}|d{1,4}|H{1,2}|h{1,2}|a|A|m{1,2}|s{1,2}|Z{1,2}|SSS/g,F=31536e6,x=2628e6,L=/^(-|\+)?P(?:([-+]?[0-9,.]*)Y)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)W)?(?:([-+]?[0-9,.]*)D)?(?:T(?:([-+]?[0-9,.]*)H)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)S)?)?$/,V={years:F,months:x,days:_,hours:y,minutes:k,seconds:a,milliseconds:1,weeks:6048e5},N=o(function(E){return E instanceof j},"c"),z=o(function(E,T,d){return new j(E,d,T.$l)},"f"),S=o(function(E){return i.p(E)+"s"},"m"),D=o(function(E){return E<0},"l"),M=o(function(E){return D(E)?Math.ceil(E):Math.floor(E)},"$"),O=o(function(E){return Math.abs(E)},"y"),A=o(function(E,T){return E?D(E)?{negative:!0,format:""+O(E)+T}:{negative:!1,format:""+E+T}:{negative:!1,format:""}},"v"),j=(function(){function E(d,u,v){var p=this;if(this.$d={},this.$l=v,d===void 0&&(this.$ms=0,this.parseFromMilliseconds()),u)return z(d*V[S(u)],this);if(typeof d=="number")return this.$ms=d,this.parseFromMilliseconds(),this;if(typeof d=="object")return Object.keys(d).forEach((function(c){p.$d[S(c)]=d[c]})),this.calMilliseconds(),this;if(typeof d=="string"){var g=d.match(L);if(g){var f=g.slice(2).map((function(c){return c!=null?Number(c):0}));return this.$d.years=f[0],this.$d.months=f[1],this.$d.weeks=f[2],this.$d.days=f[3],this.$d.hours=f[4],this.$d.minutes=f[5],this.$d.seconds=f[6],this.calMilliseconds(),this}}return this}o(E,"l");var T=E.prototype;return T.calMilliseconds=function(){var d=this;this.$ms=Object.keys(this.$d).reduce((function(u,v){return u+(d.$d[v]||0)*V[v]}),0)},T.parseFromMilliseconds=function(){var d=this.$ms;this.$d.years=M(d/F),d%=F,this.$d.months=M(d/x),d%=x,this.$d.days=M(d/_),d%=_,this.$d.hours=M(d/y),d%=y,this.$d.minutes=M(d/k),d%=k,this.$d.seconds=M(d/a),d%=a,this.$d.milliseconds=d},T.toISOString=function(){var d=A(this.$d.years,"Y"),u=A(this.$d.months,"M"),v=+this.$d.days||0;this.$d.weeks&&(v+=7*this.$d.weeks);var p=A(v,"D"),g=A(this.$d.hours,"H"),f=A(this.$d.minutes,"M"),c=this.$d.seconds||0;this.$d.milliseconds&&(c+=this.$d.milliseconds/1e3,c=Math.round(1e3*c)/1e3);var l=A(c,"S"),h=d.negative||u.negative||p.negative||g.negative||f.negative||l.negative,m=g.format||f.format||l.format?"T":"",b=(h?"-":"")+"P"+d.format+u.format+p.format+m+g.format+f.format+l.format;return b==="P"||b==="-P"?"P0D":b},T.toJSON=function(){return this.toISOString()},T.format=function(d){var u=d||"YYYY-MM-DDTHH:mm:ss",v={Y:this.$d.years,YY:i.s(this.$d.years,2,"0"),YYYY:i.s(this.$d.years,4,"0"),M:this.$d.months,MM:i.s(this.$d.months,2,"0"),D:this.$d.days,DD:i.s(this.$d.days,2,"0"),H:this.$d.hours,HH:i.s(this.$d.hours,2,"0"),m:this.$d.minutes,mm:i.s(this.$d.minutes,2,"0"),s:this.$d.seconds,ss:i.s(this.$d.seconds,2,"0"),SSS:i.s(this.$d.milliseconds,3,"0")};return u.replace(W,(function(p,g){return g||String(v[p])}))},T.as=function(d){return this.$ms/V[S(d)]},T.get=function(d){var u=this.$ms,v=S(d);return v==="milliseconds"?u%=1e3:u=v==="weeks"?M(u/V[v]):this.$d[v],u||0},T.add=function(d,u,v){var p;return p=u?d*V[S(u)]:N(d)?d.$ms:z(d,this).$ms,z(this.$ms+p*(v?-1:1),this)},T.subtract=function(d,u){return this.add(d,u,!0)},T.locale=function(d){var u=this.clone();return u.$l=d,u},T.clone=function(){return z(this.$ms,this)},T.humanize=function(d){return n().add(this.$ms,"ms").locale(this.$l).fromNow(!d)},T.valueOf=function(){return this.asMilliseconds()},T.milliseconds=function(){return this.get("milliseconds")},T.asMilliseconds=function(){return this.as("milliseconds")},T.seconds=function(){return this.get("seconds")},T.asSeconds=function(){return this.as("seconds")},T.minutes=function(){return this.get("minutes")},T.asMinutes=function(){return this.as("minutes")},T.hours=function(){return this.get("hours")},T.asHours=function(){return this.as("hours")},T.days=function(){return this.get("days")},T.asDays=function(){return this.as("days")},T.weeks=function(){return this.get("weeks")},T.asWeeks=function(){return this.as("weeks")},T.months=function(){return this.get("months")},T.asMonths=function(){return this.as("months")},T.years=function(){return this.get("years")},T.asYears=function(){return this.as("years")},E})(),B=o(function(E,T,d){return E.add(T.years()*d,"y").add(T.months()*d,"M").add(T.days()*d,"d").add(T.hours()*d,"h").add(T.minutes()*d,"m").add(T.seconds()*d,"s").add(T.milliseconds()*d,"ms")},"p");return function(E,T,d){n=d,i=d().$utils(),d.duration=function(p,g){var f=d.locale();return z(p,{$l:f},g)},d.isDuration=N;var u=T.prototype.add,v=T.prototype.subtract;T.prototype.add=function(p,g){return N(p)?B(this,p,1):u.bind(this)(p,g)},T.prototype.subtract=function(p,g){return N(p)?B(this,p,-1):v.bind(this)(p,g)}}}))}}),Ct=(function(){var t=o(function(f,c,l,h){for(l=l||{},h=f.length;h--;l[f[h]]=c);return l},"o"),r=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],n=[1,26],i=[1,27],a=[1,28],k=[1,29],y=[1,30],_=[1,31],W=[1,32],F=[1,33],x=[1,34],L=[1,9],V=[1,10],N=[1,11],z=[1,12],S=[1,13],D=[1,14],M=[1,15],O=[1,16],A=[1,19],j=[1,20],B=[1,21],E=[1,22],T=[1,23],d=[1,25],u=[1,35],v={trace:o(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:o(function(c,l,h,m,b,s,P){var e=s.length-1;switch(b){case 1:return s[e-1];case 2:this.$=[];break;case 3:s[e-1].push(s[e]),this.$=s[e-1];break;case 4:case 5:this.$=s[e];break;case 6:case 7:this.$=[];break;case 8:m.setWeekday("monday");break;case 9:m.setWeekday("tuesday");break;case 10:m.setWeekday("wednesday");break;case 11:m.setWeekday("thursday");break;case 12:m.setWeekday("friday");break;case 13:m.setWeekday("saturday");break;case 14:m.setWeekday("sunday");break;case 15:m.setWeekend("friday");break;case 16:m.setWeekend("saturday");break;case 17:m.setDateFormat(s[e].substr(11)),this.$=s[e].substr(11);break;case 18:m.enableInclusiveEndDates(),this.$=s[e].substr(18);break;case 19:m.TopAxis(),this.$=s[e].substr(8);break;case 20:m.setAxisFormat(s[e].substr(11)),this.$=s[e].substr(11);break;case 21:m.setTickInterval(s[e].substr(13)),this.$=s[e].substr(13);break;case 22:m.setExcludes(s[e].substr(9)),this.$=s[e].substr(9);break;case 23:m.setIncludes(s[e].substr(9)),this.$=s[e].substr(9);break;case 24:m.setTodayMarker(s[e].substr(12)),this.$=s[e].substr(12);break;case 27:m.setDiagramTitle(s[e].substr(6)),this.$=s[e].substr(6);break;case 28:this.$=s[e].trim(),m.setAccTitle(this.$);break;case 29:case 30:this.$=s[e].trim(),m.setAccDescription(this.$);break;case 31:m.addSection(s[e].substr(8)),this.$=s[e].substr(8);break;case 33:m.addTask(s[e-1],s[e]),this.$="task";break;case 34:this.$=s[e-1],m.setClickEvent(s[e-1],s[e],null);break;case 35:this.$=s[e-2],m.setClickEvent(s[e-2],s[e-1],s[e]);break;case 36:this.$=s[e-2],m.setClickEvent(s[e-2],s[e-1],null),m.setLink(s[e-2],s[e]);break;case 37:this.$=s[e-3],m.setClickEvent(s[e-3],s[e-2],s[e-1]),m.setLink(s[e-3],s[e]);break;case 38:this.$=s[e-2],m.setClickEvent(s[e-2],s[e],null),m.setLink(s[e-2],s[e-1]);break;case 39:this.$=s[e-3],m.setClickEvent(s[e-3],s[e-1],s[e]),m.setLink(s[e-3],s[e-2]);break;case 40:this.$=s[e-1],m.setLink(s[e-1],s[e]);break;case 41:case 47:this.$=s[e-1]+" "+s[e];break;case 42:case 43:case 45:this.$=s[e-2]+" "+s[e-1]+" "+s[e];break;case 44:case 46:this.$=s[e-3]+" "+s[e-2]+" "+s[e-1]+" "+s[e];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(r,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:n,13:i,14:a,15:k,16:y,17:_,18:W,19:18,20:F,21:x,22:L,23:V,24:N,25:z,26:S,27:D,28:M,29:O,30:A,31:j,33:B,35:E,36:T,37:24,38:d,40:u},t(r,[2,7],{1:[2,1]}),t(r,[2,3]),{9:36,11:17,12:n,13:i,14:a,15:k,16:y,17:_,18:W,19:18,20:F,21:x,22:L,23:V,24:N,25:z,26:S,27:D,28:M,29:O,30:A,31:j,33:B,35:E,36:T,37:24,38:d,40:u},t(r,[2,5]),t(r,[2,6]),t(r,[2,17]),t(r,[2,18]),t(r,[2,19]),t(r,[2,20]),t(r,[2,21]),t(r,[2,22]),t(r,[2,23]),t(r,[2,24]),t(r,[2,25]),t(r,[2,26]),t(r,[2,27]),{32:[1,37]},{34:[1,38]},t(r,[2,30]),t(r,[2,31]),t(r,[2,32]),{39:[1,39]},t(r,[2,8]),t(r,[2,9]),t(r,[2,10]),t(r,[2,11]),t(r,[2,12]),t(r,[2,13]),t(r,[2,14]),t(r,[2,15]),t(r,[2,16]),{41:[1,40],43:[1,41]},t(r,[2,4]),t(r,[2,28]),t(r,[2,29]),t(r,[2,33]),t(r,[2,34],{42:[1,42],43:[1,43]}),t(r,[2,40],{41:[1,44]}),t(r,[2,35],{43:[1,45]}),t(r,[2,36]),t(r,[2,38],{42:[1,46]}),t(r,[2,37]),t(r,[2,39])],defaultActions:{},parseError:o(function(c,l){if(l.recoverable)this.trace(c);else{var h=new Error(c);throw h.hash=l,h}},"parseError"),parse:o(function(c){var l=this,h=[0],m=[],b=[null],s=[],P=this.table,e="",w=0,$=0,Y=2,I=1,q=s.slice.call(arguments,1),C=Object.create(this.lexer),Q={yy:{}};for(var st in this.yy)Object.prototype.hasOwnProperty.call(this.yy,st)&&(Q.yy[st]=this.yy[st]);C.setInput(c,Q.yy),Q.yy.lexer=C,Q.yy.parser=this,typeof C.yylloc>"u"&&(C.yylloc={});var ct=C.yylloc;s.push(ct);var ht=C.options&&C.options.ranges;typeof Q.yy.parseError=="function"?this.parseError=Q.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function mt(X){h.length=h.length-2*X,b.length=b.length-X,s.length=s.length-X}o(mt,"popStack");function lt(){var X;return X=m.pop()||C.lex()||I,typeof X!="number"&&(X instanceof Array&&(m=X,X=m.pop()),X=l.symbols_[X]||X),X}o(lt,"lex");for(var R,U,G,nt,J={},it,K,jt,vt;;){if(U=h[h.length-1],this.defaultActions[U]?G=this.defaultActions[U]:((R===null||typeof R>"u")&&(R=lt()),G=P[U]&&P[U][R]),typeof G>"u"||!G.length||!G[0]){var St="";vt=[];for(it in P[U])this.terminals_[it]&&it>Y&&vt.push("'"+this.terminals_[it]+"'");C.showPosition?St="Parse error on line "+(w+1)+`:
`+C.showPosition()+`
Expecting `+vt.join(", ")+", got '"+(this.terminals_[R]||R)+"'":St="Parse error on line "+(w+1)+": Unexpected "+(R==I?"end of input":"'"+(this.terminals_[R]||R)+"'"),this.parseError(St,{text:C.match,token:this.terminals_[R]||R,line:C.yylineno,loc:ct,expected:vt})}if(G[0]instanceof Array&&G.length>1)throw new Error("Parse Error: multiple actions possible at state: "+U+", token: "+R);switch(G[0]){case 1:h.push(R),b.push(C.yytext),s.push(C.yylloc),h.push(G[1]),R=null,$=C.yyleng,e=C.yytext,w=C.yylineno,ct=C.yylloc;break;case 2:if(K=this.productions_[G[1]][1],J.$=b[b.length-K],J._$={first_line:s[s.length-(K||1)].first_line,last_line:s[s.length-1].last_line,first_column:s[s.length-(K||1)].first_column,last_column:s[s.length-1].last_column},ht&&(J._$.range=[s[s.length-(K||1)].range[0],s[s.length-1].range[1]]),nt=this.performAction.apply(J,[e,$,w,Q.yy,G[1],b,s].concat(q)),typeof nt<"u")return nt;K&&(h=h.slice(0,-1*K*2),b=b.slice(0,-1*K),s=s.slice(0,-1*K)),h.push(this.productions_[G[1]][0]),b.push(J.$),s.push(J._$),jt=P[h[h.length-2]][h[h.length-1]],h.push(jt);break;case 3:return!0}}return!0},"parse")},p=(function(){var f={EOF:1,parseError:o(function(l,h){if(this.yy.parser)this.yy.parser.parseError(l,h);else throw new Error(l)},"parseError"),setInput:o(function(c,l){return this.yy=l||this.yy||{},this._input=c,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:o(function(){var c=this._input[0];this.yytext+=c,this.yyleng++,this.offset++,this.match+=c,this.matched+=c;var l=c.match(/(?:\r\n?|\n).*/g);return l?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),c},"input"),unput:o(function(c){var l=c.length,h=c.split(/(?:\r\n?|\n)/g);this._input=c+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-l),this.offset-=l;var m=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),h.length-1&&(this.yylineno-=h.length-1);var b=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:h?(h.length===m.length?this.yylloc.first_column:0)+m[m.length-h.length].length-h[0].length:this.yylloc.first_column-l},this.options.ranges&&(this.yylloc.range=[b[0],b[0]+this.yyleng-l]),this.yyleng=this.yytext.length,this},"unput"),more:o(function(){return this._more=!0,this},"more"),reject:o(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:o(function(c){this.unput(this.match.slice(c))},"less"),pastInput:o(function(){var c=this.matched.substr(0,this.matched.length-this.match.length);return(c.length>20?"...":"")+c.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:o(function(){var c=this.match;return c.length<20&&(c+=this._input.substr(0,20-c.length)),(c.substr(0,20)+(c.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:o(function(){var c=this.pastInput(),l=new Array(c.length+1).join("-");return c+this.upcomingInput()+`
`+l+"^"},"showPosition"),test_match:o(function(c,l){var h,m,b;if(this.options.backtrack_lexer&&(b={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(b.yylloc.range=this.yylloc.range.slice(0))),m=c[0].match(/(?:\r\n?|\n).*/g),m&&(this.yylineno+=m.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:m?m[m.length-1].length-m[m.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+c[0].length},this.yytext+=c[0],this.match+=c[0],this.matches=c,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(c[0].length),this.matched+=c[0],h=this.performAction.call(this,this.yy,this,l,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),h)return h;if(this._backtrack){for(var s in b)this[s]=b[s];return!1}return!1},"test_match"),next:o(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var c,l,h,m;this._more||(this.yytext="",this.match="");for(var b=this._currentRules(),s=0;s<b.length;s++)if(h=this._input.match(this.rules[b[s]]),h&&(!l||h[0].length>l[0].length)){if(l=h,m=s,this.options.backtrack_lexer){if(c=this.test_match(h,b[s]),c!==!1)return c;if(this._backtrack){l=!1;continue}else return!1}else if(!this.options.flex)break}return l?(c=this.test_match(l,b[m]),c!==!1?c:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:o(function(){var l=this.next();return l||this.lex()},"lex"),begin:o(function(l){this.conditionStack.push(l)},"begin"),popState:o(function(){var l=this.conditionStack.length-1;return l>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:o(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:o(function(l){return l=this.conditionStack.length-1-Math.abs(l||0),l>=0?this.conditionStack[l]:"INITIAL"},"topState"),pushState:o(function(l){this.begin(l)},"pushState"),stateStackSize:o(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:o(function(l,h,m,b){switch(m){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}};return f})();v.lexer=p;function g(){this.yy={}}return o(g,"Parser"),g.prototype=v,v.Parser=g,new g})();Ct.parser=Ct;var Oe=Ct,Pe=at(Ye()),Z=at(Kt()),Ve=at($e()),Ne=at(Fe()),ze=at(Le());Z.default.extend(Ve.default);Z.default.extend(Ne.default);Z.default.extend(ze.default);var Zt={friday:5,saturday:6},tt="",At="",$t=void 0,Ft="",yt=[],gt=[],Lt=new Map,Wt=[],wt=[],ft="",Ot="",te=["active","done","crit","milestone","vert"],Pt=[],pt=!1,Vt=!1,Nt="sunday",_t="saturday",Et=0,je=o(function(){Wt=[],wt=[],ft="",Pt=[],bt=0,Yt=void 0,xt=void 0,H=[],tt="",At="",Ot="",$t=void 0,Ft="",yt=[],gt=[],pt=!1,Vt=!1,Et=0,Lt=new Map,Ie(),Nt="sunday",_t="saturday"},"clear"),Re=o(function(t){At=t},"setAxisFormat"),He=o(function(){return At},"getAxisFormat"),Be=o(function(t){$t=t},"setTickInterval"),qe=o(function(){return $t},"getTickInterval"),Ge=o(function(t){Ft=t},"setTodayMarker"),Xe=o(function(){return Ft},"getTodayMarker"),Ue=o(function(t){tt=t},"setDateFormat"),Ze=o(function(){pt=!0},"enableInclusiveEndDates"),Qe=o(function(){return pt},"endDatesAreInclusive"),Je=o(function(){Vt=!0},"enableTopAxis"),Ke=o(function(){return Vt},"topAxisEnabled"),ts=o(function(t){Ot=t},"setDisplayMode"),es=o(function(){return Ot},"getDisplayMode"),ss=o(function(){return tt},"getDateFormat"),is=o(function(t){yt=t.toLowerCase().split(/[\s,]+/)},"setIncludes"),ns=o(function(){return yt},"getIncludes"),rs=o(function(t){gt=t.toLowerCase().split(/[\s,]+/)},"setExcludes"),as=o(function(){return gt},"getExcludes"),os=o(function(){return Lt},"getLinks"),cs=o(function(t){ft=t,Wt.push(t)},"addSection"),ls=o(function(){return Wt},"getSections"),us=o(function(){let t=Qt();const r=10;let n=0;for(;!t&&n<r;)t=Qt(),n++;return wt=H,wt},"getTasks"),ee=o(function(t,r,n,i){const a=t.format(r.trim()),k=t.format("YYYY-MM-DD");return i.includes(a)||i.includes(k)?!1:n.includes("weekends")&&(t.isoWeekday()===Zt[_t]||t.isoWeekday()===Zt[_t]+1)||n.includes(t.format("dddd").toLowerCase())?!0:n.includes(a)||n.includes(k)},"isInvalidDate"),ds=o(function(t){Nt=t},"setWeekday"),fs=o(function(){return Nt},"getWeekday"),hs=o(function(t){_t=t},"setWeekend"),se=o(function(t,r,n,i){if(!n.length||t.manualEndTime)return;let a;t.startTime instanceof Date?a=(0,Z.default)(t.startTime):a=(0,Z.default)(t.startTime,r,!0),a=a.add(1,"d");let k;t.endTime instanceof Date?k=(0,Z.default)(t.endTime):k=(0,Z.default)(t.endTime,r,!0);const[y,_]=ms(a,k,r,n,i);t.endTime=y.toDate(),t.renderEndTime=_},"checkTaskDates"),ms=o(function(t,r,n,i,a){let k=!1,y=null;for(;t<=r;)k||(y=r.toDate()),k=ee(t,n,i,a),k&&(r=r.add(1,"d")),t=t.add(1,"d");return[r,y]},"fixTaskDates"),It=o(function(t,r,n){if(n=n.trim(),o(_=>{const W=_.trim();return W==="x"||W==="X"},"isTimestampFormat")(r)&&/^\d+$/.test(n))return new Date(Number(n));const k=/^after\s+(?<ids>[\d\w- ]+)/.exec(n);if(k!==null){let _=null;for(const F of k.groups.ids.split(" ")){let x=ot(F);x!==void 0&&(!_||x.endTime>_.endTime)&&(_=x)}if(_)return _.endTime;const W=new Date;return W.setHours(0,0,0,0),W}let y=(0,Z.default)(n,r.trim(),!0);if(y.isValid())return y.toDate();{rt.debug("Invalid date:"+n),rt.debug("With date format:"+r.trim());const _=new Date(n);if(_===void 0||isNaN(_.getTime())||_.getFullYear()<-1e4||_.getFullYear()>1e4)throw new Error("Invalid date:"+n);return _}},"getStartDate"),ie=o(function(t){const r=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return r!==null?[Number.parseFloat(r[1]),r[2]]:[NaN,"ms"]},"parseDuration"),ne=o(function(t,r,n,i=!1){n=n.trim();const k=/^until\s+(?<ids>[\d\w- ]+)/.exec(n);if(k!==null){let x=null;for(const V of k.groups.ids.split(" ")){let N=ot(V);N!==void 0&&(!x||N.startTime<x.startTime)&&(x=N)}if(x)return x.startTime;const L=new Date;return L.setHours(0,0,0,0),L}let y=(0,Z.default)(n,r.trim(),!0);if(y.isValid())return i&&(y=y.add(1,"d")),y.toDate();let _=(0,Z.default)(t);const[W,F]=ie(n);if(!Number.isNaN(W)){const x=_.add(W,F);x.isValid()&&(_=x)}return _.toDate()},"getEndDate"),bt=0,dt=o(function(t){return t===void 0?(bt=bt+1,"task"+bt):t},"parseId"),ks=o(function(t,r){let n;r.substr(0,1)===":"?n=r.substr(1,r.length):n=r;const i=n.split(","),a={};zt(i,a,te);for(let y=0;y<i.length;y++)i[y]=i[y].trim();let k="";switch(i.length){case 1:a.id=dt(),a.startTime=t.endTime,k=i[0];break;case 2:a.id=dt(),a.startTime=It(void 0,tt,i[0]),k=i[1];break;case 3:a.id=dt(i[0]),a.startTime=It(void 0,tt,i[1]),k=i[2];break}return k&&(a.endTime=ne(a.startTime,tt,k,pt),a.manualEndTime=(0,Z.default)(k,"YYYY-MM-DD",!0).isValid(),se(a,tt,gt,yt)),a},"compileData"),ys=o(function(t,r){let n;r.substr(0,1)===":"?n=r.substr(1,r.length):n=r;const i=n.split(","),a={};zt(i,a,te);for(let k=0;k<i.length;k++)i[k]=i[k].trim();switch(i.length){case 1:a.id=dt(),a.startTime={type:"prevTaskEnd",id:t},a.endTime={data:i[0]};break;case 2:a.id=dt(),a.startTime={type:"getStartDate",startData:i[0]},a.endTime={data:i[1]};break;case 3:a.id=dt(i[0]),a.startTime={type:"getStartDate",startData:i[1]},a.endTime={data:i[2]};break}return a},"parseData"),Yt,xt,H=[],re={},gs=o(function(t,r){const n={section:ft,type:ft,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:r},task:t,classes:[]},i=ys(xt,r);n.raw.startTime=i.startTime,n.raw.endTime=i.endTime,n.id=i.id,n.prevTaskId=xt,n.active=i.active,n.done=i.done,n.crit=i.crit,n.milestone=i.milestone,n.vert=i.vert,n.order=Et,Et++;const a=H.push(n);xt=n.id,re[n.id]=a-1},"addTask"),ot=o(function(t){const r=re[t];return H[r]},"findTaskById"),ps=o(function(t,r){const n={section:ft,type:ft,description:t,task:t,classes:[]},i=ks(Yt,r);n.startTime=i.startTime,n.endTime=i.endTime,n.id=i.id,n.active=i.active,n.done=i.done,n.crit=i.crit,n.milestone=i.milestone,n.vert=i.vert,Yt=n,wt.push(n)},"addTaskOrg"),Qt=o(function(){const t=o(function(n){const i=H[n];let a="";switch(H[n].raw.startTime.type){case"prevTaskEnd":{const k=ot(i.prevTaskId);i.startTime=k.endTime;break}case"getStartDate":a=It(void 0,tt,H[n].raw.startTime.startData),a&&(H[n].startTime=a);break}return H[n].startTime&&(H[n].endTime=ne(H[n].startTime,tt,H[n].raw.endTime.data,pt),H[n].endTime&&(H[n].processed=!0,H[n].manualEndTime=(0,Z.default)(H[n].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),se(H[n],tt,gt,yt))),H[n].processed},"compileTask");let r=!0;for(const[n,i]of H.entries())t(n),r=r&&i.processed;return r},"compileTasks"),vs=o(function(t,r){let n=r;ut().securityLevel!=="loose"&&(n=(0,Pe.sanitizeUrl)(r)),t.split(",").forEach(function(i){ot(i)!==void 0&&(oe(i,()=>{window.open(n,"_self")}),Lt.set(i,n))}),ae(t,"clickable")},"setLink"),ae=o(function(t,r){t.split(",").forEach(function(n){let i=ot(n);i!==void 0&&i.classes.push(r)})},"setClass"),Ts=o(function(t,r,n){if(ut().securityLevel!=="loose"||r===void 0)return;let i=[];if(typeof n=="string"){i=n.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let k=0;k<i.length;k++){let y=i[k].trim();y.startsWith('"')&&y.endsWith('"')&&(y=y.substr(1,y.length-2)),i[k]=y}}i.length===0&&i.push(t),ot(t)!==void 0&&oe(t,()=>{Ae.runFunc(r,...i)})},"setClickFun"),oe=o(function(t,r){Pt.push(function(){const n=document.querySelector(`[id="${t}"]`);n!==null&&n.addEventListener("click",function(){r()})},function(){const n=document.querySelector(`[id="${t}-text"]`);n!==null&&n.addEventListener("click",function(){r()})})},"pushFun"),bs=o(function(t,r,n){t.split(",").forEach(function(i){Ts(i,r,n)}),ae(t,"clickable")},"setClickEvent"),xs=o(function(t){Pt.forEach(function(r){r(t)})},"bindFunctions"),ws={getConfig:o(()=>ut().gantt,"getConfig"),clear:je,setDateFormat:Ue,getDateFormat:ss,enableInclusiveEndDates:Ze,endDatesAreInclusive:Qe,enableTopAxis:Je,topAxisEnabled:Ke,setAxisFormat:Re,getAxisFormat:He,setTickInterval:Be,getTickInterval:qe,setTodayMarker:Ge,getTodayMarker:Xe,setAccTitle:he,getAccTitle:fe,setDiagramTitle:de,getDiagramTitle:ue,setDisplayMode:ts,getDisplayMode:es,setAccDescription:le,getAccDescription:ce,addSection:cs,getSections:ls,getTasks:us,addTask:gs,findTaskById:ot,addTaskOrg:ps,setIncludes:is,getIncludes:ns,setExcludes:rs,getExcludes:as,setClickEvent:bs,setLink:vs,getLinks:os,bindFunctions:xs,parseDuration:ie,isInvalidDate:ee,setWeekday:ds,getWeekday:fs,setWeekend:hs};function zt(t,r,n){let i=!0;for(;i;)i=!1,n.forEach(function(a){const k="^\\s*"+a+"\\s*$",y=new RegExp(k);t[0].match(y)&&(r[a]=!0,t.shift(1),i=!0)})}o(zt,"getTaskTags");var kt=at(Kt()),_s=at(We());kt.default.extend(_s.default);var Ds=o(function(){rt.debug("Something is calling, setConf, remove the call")},"setConf"),Jt={monday:Me,tuesday:Se,wednesday:De,thursday:_e,friday:we,saturday:xe,sunday:be},Ss=o((t,r)=>{let n=[...t].map(()=>-1/0),i=[...t].sort((k,y)=>k.startTime-y.startTime||k.order-y.order),a=0;for(const k of i)for(let y=0;y<n.length;y++)if(k.startTime>=n[y]){n[y]=k.endTime,k.order=y+r,y>a&&(a=y);break}return a},"getMaxIntersections"),et,Mt=1e4,Ms=o(function(t,r,n,i){const a=ut().gantt,k=ut().securityLevel;let y;k==="sandbox"&&(y=Tt("#i"+r));const _=k==="sandbox"?Tt(y.nodes()[0].contentDocument.body):Tt("body"),W=k==="sandbox"?y.nodes()[0].contentDocument:document,F=W.getElementById(r);et=F.parentElement.offsetWidth,et===void 0&&(et=1200),a.useWidth!==void 0&&(et=a.useWidth);const x=i.db.getTasks();let L=[];for(const u of x)L.push(u.type);L=d(L);const V={};let N=2*a.topPadding;if(i.db.getDisplayMode()==="compact"||a.displayMode==="compact"){const u={};for(const p of x)u[p.section]===void 0?u[p.section]=[p]:u[p.section].push(p);let v=0;for(const p of Object.keys(u)){const g=Ss(u[p],v)+1;v+=g,N+=g*(a.barHeight+a.barGap),V[p]=g}}else{N+=x.length*(a.barHeight+a.barGap);for(const u of L)V[u]=x.filter(v=>v.type===u).length}F.setAttribute("viewBox","0 0 "+et+" "+N);const z=_.select(`[id="${r}"]`),S=me().domain([ke(x,function(u){return u.startTime}),ye(x,function(u){return u.endTime})]).rangeRound([0,et-a.leftPadding-a.rightPadding]);function D(u,v){const p=u.startTime,g=v.startTime;let f=0;return p>g?f=1:p<g&&(f=-1),f}o(D,"taskCompare"),x.sort(D),M(x,et,N),ge(z,N,et,a.useMaxWidth),z.append("text").text(i.db.getDiagramTitle()).attr("x",et/2).attr("y",a.titleTopMargin).attr("class","titleText");function M(u,v,p){const g=a.barHeight,f=g+a.barGap,c=a.topPadding,l=a.leftPadding,h=pe().domain([0,L.length]).range(["#00B9FA","#F95002"]).interpolate(ve);A(f,c,l,v,p,u,i.db.getExcludes(),i.db.getIncludes()),B(l,c,v,p),O(u,f,c,l,g,h,v),E(f,c),T(l,c,v,p)}o(M,"makeGantt");function O(u,v,p,g,f,c,l){u.sort((e,w)=>e.vert===w.vert?0:e.vert?1:-1);const m=[...new Set(u.map(e=>e.order))].map(e=>u.find(w=>w.order===e));z.append("g").selectAll("rect").data(m).enter().append("rect").attr("x",0).attr("y",function(e,w){return w=e.order,w*v+p-2}).attr("width",function(){return l-a.rightPadding/2}).attr("height",v).attr("class",function(e){for(const[w,$]of L.entries())if(e.type===$)return"section section"+w%a.numberSectionStyles;return"section section0"}).enter();const b=z.append("g").selectAll("rect").data(u).enter(),s=i.db.getLinks();if(b.append("rect").attr("id",function(e){return e.id}).attr("rx",3).attr("ry",3).attr("x",function(e){return e.milestone?S(e.startTime)+g+.5*(S(e.endTime)-S(e.startTime))-.5*f:S(e.startTime)+g}).attr("y",function(e,w){return w=e.order,e.vert?a.gridLineStartPadding:w*v+p}).attr("width",function(e){return e.milestone?f:e.vert?.08*f:S(e.renderEndTime||e.endTime)-S(e.startTime)}).attr("height",function(e){return e.vert?x.length*(a.barHeight+a.barGap)+a.barHeight*2:f}).attr("transform-origin",function(e,w){return w=e.order,(S(e.startTime)+g+.5*(S(e.endTime)-S(e.startTime))).toString()+"px "+(w*v+p+.5*f).toString()+"px"}).attr("class",function(e){const w="task";let $="";e.classes.length>0&&($=e.classes.join(" "));let Y=0;for(const[q,C]of L.entries())e.type===C&&(Y=q%a.numberSectionStyles);let I="";return e.active?e.crit?I+=" activeCrit":I=" active":e.done?e.crit?I=" doneCrit":I=" done":e.crit&&(I+=" crit"),I.length===0&&(I=" task"),e.milestone&&(I=" milestone "+I),e.vert&&(I=" vert "+I),I+=Y,I+=" "+$,w+I}),b.append("text").attr("id",function(e){return e.id+"-text"}).text(function(e){return e.task}).attr("font-size",a.fontSize).attr("x",function(e){let w=S(e.startTime),$=S(e.renderEndTime||e.endTime);if(e.milestone&&(w+=.5*(S(e.endTime)-S(e.startTime))-.5*f,$=w+f),e.vert)return S(e.startTime)+g;const Y=this.getBBox().width;return Y>$-w?$+Y+1.5*a.leftPadding>l?w+g-5:$+g+5:($-w)/2+w+g}).attr("y",function(e,w){return e.vert?a.gridLineStartPadding+x.length*(a.barHeight+a.barGap)+60:(w=e.order,w*v+a.barHeight/2+(a.fontSize/2-2)+p)}).attr("text-height",f).attr("class",function(e){const w=S(e.startTime);let $=S(e.endTime);e.milestone&&($=w+f);const Y=this.getBBox().width;let I="";e.classes.length>0&&(I=e.classes.join(" "));let q=0;for(const[Q,st]of L.entries())e.type===st&&(q=Q%a.numberSectionStyles);let C="";return e.active&&(e.crit?C="activeCritText"+q:C="activeText"+q),e.done?e.crit?C=C+" doneCritText"+q:C=C+" doneText"+q:e.crit&&(C=C+" critText"+q),e.milestone&&(C+=" milestoneText"),e.vert&&(C+=" vertText"),Y>$-w?$+Y+1.5*a.leftPadding>l?I+" taskTextOutsideLeft taskTextOutside"+q+" "+C:I+" taskTextOutsideRight taskTextOutside"+q+" "+C+" width-"+Y:I+" taskText taskText"+q+" "+C+" width-"+Y}),ut().securityLevel==="sandbox"){let e;e=Tt("#i"+r);const w=e.nodes()[0].contentDocument;b.filter(function($){return s.has($.id)}).each(function($){var Y=w.querySelector("#"+$.id),I=w.querySelector("#"+$.id+"-text");const q=Y.parentNode;var C=w.createElement("a");C.setAttribute("xlink:href",s.get($.id)),C.setAttribute("target","_top"),q.appendChild(C),C.appendChild(Y),C.appendChild(I)})}}o(O,"drawRects");function A(u,v,p,g,f,c,l,h){if(l.length===0&&h.length===0)return;let m,b;for(const{startTime:Y,endTime:I}of c)(m===void 0||Y<m)&&(m=Y),(b===void 0||I>b)&&(b=I);if(!m||!b)return;if((0,kt.default)(b).diff((0,kt.default)(m),"year")>5){rt.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}const s=i.db.getDateFormat(),P=[];let e=null,w=(0,kt.default)(m);for(;w.valueOf()<=b;)i.db.isInvalidDate(w,s,l,h)?e?e.end=w:e={start:w,end:w}:e&&(P.push(e),e=null),w=w.add(1,"d");z.append("g").selectAll("rect").data(P).enter().append("rect").attr("id",Y=>"exclude-"+Y.start.format("YYYY-MM-DD")).attr("x",Y=>S(Y.start.startOf("day"))+p).attr("y",a.gridLineStartPadding).attr("width",Y=>S(Y.end.endOf("day"))-S(Y.start.startOf("day"))).attr("height",f-v-a.gridLineStartPadding).attr("transform-origin",function(Y,I){return(S(Y.start)+p+.5*(S(Y.end)-S(Y.start))).toString()+"px "+(I*u+.5*f).toString()+"px"}).attr("class","exclude-range")}o(A,"drawExcludeDays");function j(u,v,p,g){if(p<=0||u>v)return 1/0;const f=v-u,c=kt.default.duration({[g??"day"]:p}).asMilliseconds();return c<=0?1/0:Math.ceil(f/c)}o(j,"getEstimatedTickCount");function B(u,v,p,g){const f=i.db.getDateFormat(),c=i.db.getAxisFormat();let l;c?l=c:f==="D"?l="%d":l=a.axisFormat??"%Y-%m-%d";let h=Te(S).tickSize(-g+v+a.gridLineStartPadding).tickFormat(Rt(l));const b=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(i.db.getTickInterval()||a.tickInterval);if(b!==null){const s=parseInt(b[1],10);if(isNaN(s)||s<=0)rt.warn(`Invalid tick interval value: "${b[1]}". Skipping custom tick interval.`);else{const P=b[2],e=i.db.getWeekday()||a.weekday,w=S.domain(),$=w[0],Y=w[1],I=j($,Y,s,P);if(I>Mt)rt.warn(`The tick interval "${s}${P}" would generate ${I} ticks, which exceeds the maximum allowed (${Mt}). This may indicate an invalid date or time range. Skipping custom tick interval.`);else switch(P){case"millisecond":h.ticks(Ut.every(s));break;case"second":h.ticks(Xt.every(s));break;case"minute":h.ticks(Gt.every(s));break;case"hour":h.ticks(qt.every(s));break;case"day":h.ticks(Bt.every(s));break;case"week":h.ticks(Jt[e].every(s));break;case"month":h.ticks(Ht.every(s));break}}}if(z.append("g").attr("class","grid").attr("transform","translate("+u+", "+(g-50)+")").call(h).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),i.db.topAxisEnabled()||a.topAxis){let s=Ce(S).tickSize(-g+v+a.gridLineStartPadding).tickFormat(Rt(l));if(b!==null){const P=parseInt(b[1],10);if(isNaN(P)||P<=0)rt.warn(`Invalid tick interval value: "${b[1]}". Skipping custom tick interval.`);else{const e=b[2],w=i.db.getWeekday()||a.weekday,$=S.domain(),Y=$[0],I=$[1];if(j(Y,I,P,e)<=Mt)switch(e){case"millisecond":s.ticks(Ut.every(P));break;case"second":s.ticks(Xt.every(P));break;case"minute":s.ticks(Gt.every(P));break;case"hour":s.ticks(qt.every(P));break;case"day":s.ticks(Bt.every(P));break;case"week":s.ticks(Jt[w].every(P));break;case"month":s.ticks(Ht.every(P));break}}}z.append("g").attr("class","grid").attr("transform","translate("+u+", "+v+")").call(s).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}o(B,"makeGrid");function E(u,v){let p=0;const g=Object.keys(V).map(f=>[f,V[f]]);z.append("g").selectAll("text").data(g).enter().append(function(f){const c=f[0].split(Ee.lineBreakRegex),l=-(c.length-1)/2,h=W.createElementNS("http://www.w3.org/2000/svg","text");h.setAttribute("dy",l+"em");for(const[m,b]of c.entries()){const s=W.createElementNS("http://www.w3.org/2000/svg","tspan");s.setAttribute("alignment-baseline","central"),s.setAttribute("x","10"),m>0&&s.setAttribute("dy","1em"),s.textContent=b,h.appendChild(s)}return h}).attr("x",10).attr("y",function(f,c){if(c>0)for(let l=0;l<c;l++)return p+=g[c-1][1],f[1]*u/2+p*u+v;else return f[1]*u/2+v}).attr("font-size",a.sectionFontSize).attr("class",function(f){for(const[c,l]of L.entries())if(f[0]===l)return"sectionTitle sectionTitle"+c%a.numberSectionStyles;return"sectionTitle"})}o(E,"vertLabels");function T(u,v,p,g){const f=i.db.getTodayMarker();if(f==="off")return;const c=z.append("g").attr("class","today"),l=new Date,h=c.append("line");h.attr("x1",S(l)+u).attr("x2",S(l)+u).attr("y1",a.titleTopMargin).attr("y2",g-a.titleTopMargin).attr("class","today"),f!==""&&h.attr("style",f.replace(/,/g,";"))}o(T,"drawToday");function d(u){const v={},p=[];for(let g=0,f=u.length;g<f;++g)Object.prototype.hasOwnProperty.call(v,u[g])||(v[u[g]]=!0,p.push(u[g]));return p}o(d,"checkUnique")},"draw"),Cs={setConf:Ds,draw:Ms},Es=o(t=>`
  .mermaid-main-font {
        font-family: ${t.fontFamily};
  }

  .exclude-range {
    fill: ${t.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${t.sectionBkgColor};
  }

  .section2 {
    fill: ${t.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${t.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${t.titleColor};
  }

  .sectionTitle1 {
    fill: ${t.titleColor};
  }

  .sectionTitle2 {
    fill: ${t.titleColor};
  }

  .sectionTitle3 {
    fill: ${t.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${t.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${t.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${t.fontFamily};
    fill: ${t.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${t.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${t.taskTextDarkColor};
    text-anchor: start;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${t.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${t.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${t.taskBkgColor};
    stroke: ${t.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${t.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${t.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${t.activeTaskBkgColor};
    stroke: ${t.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${t.doneTaskBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
    cursor: pointer;
    shape-rendering: crispEdges;
  }

  .milestone {
    transform: rotate(45deg) scale(0.8,0.8);
  }

  .milestoneText {
    font-style: italic;
  }
  .doneCritText0,
  .doneCritText1,
  .doneCritText2,
  .doneCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  .vert {
    stroke: ${t.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${t.vertLineColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${t.titleColor||t.textColor};
    font-family: ${t.fontFamily};
  }
`,"getStyles"),Is=Es,Os={parser:Oe,db:ws,renderer:Cs,styles:Is};export{Os as diagram};
