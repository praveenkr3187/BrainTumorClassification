var getstate = $.get('/state');
getstate.done(function(state){
for(var i=0;i<34;i++){
    document.getElementById("sel2").innerHTML += '<option value="'+state.state[i]+'">'+state.state[i]+'</option>';
}
});

$(document).ready(function() {

  $("#sel2").change(function() {

    var datastate = $(this).val() ;
    document.getElementById("sel1").innerHTML="";
    console.log(datastate);
        $.ajax({
            type: "POST",
            url: "/df2",
            data: { st : datastate } 
        }).done(function(df2){
          for(var i=0;i<df2.district.length;i++){
            document.getElementById("sel1").innerHTML += '<option value="'+df2.district[i]+'">'+df2.district[i]+'</option>';
                            }
        });

  });

});

function setImage(img){
  document.getElementsByClassName("realimage").src = img;
}
